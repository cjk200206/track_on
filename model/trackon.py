import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch

from model.event_backbone import EventBackbone, build_backbone
from model.modules import MHA_Block, SimpleFPN
from model.prediction_head import Prediction_Head
from model.reranking import Rerank_Module
from model.trajectory_history import TrajectoryHistoryScorer, trajectory_entry
from model.candidate_history import CandidateHistoryScorer
from model.point_memory import PointMemory, RGBMemoryReader, read_memory, resize_time_embedding

from utils.coord_utils import indices_to_coords

class Track_On2(nn.Module):
    def __init__(self, args, nhead=4):
        super().__init__()

        # === Hyper-parameters ===
        self.K = args.K                  # Top-K regions for re-ranking                 
        self.D = args.D                  # Feature dimension    
        self.M = args.M                  # Memory size
        
        self.decoder_layer_num = args.decoder_layer_num     # Decoder layer num
        # === === ===

        # === Size related ===
        self.input_size = args.input_size
        self.stride = 4
        self.H = self.input_size[0]         # Original image size
        self.W = self.input_size[1]
        self.Hf = self.H // self.stride     # Feature map size:    
        self.Wf = self.W // self.stride
        self.P = int(self.Hf * self.Wf)     # Number of patches in the feature map:
        # === === ===

        # === Modules ===
        self.backbone = build_backbone(args)
        self.fpn = SimpleFPN(args)
        self.reranking_head = Rerank_Module(args, nhead)
        self.prediction_head = Prediction_Head(args, nhead)

        # Query Decoder Layers
        self.feature_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)
        self.query_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)
        self.memory_attention = self._make_transformer_layer(self.decoder_layer_num, nhead)

        self.projection1 = nn.Linear(args.D, args.D)
        self.projection2 = nn.Linear(args.D, args.D)
        self.ms_corr_proj = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        history_size = int(getattr(args, "trajectory_history_size", 0))
        if history_size > self.M:
            raise ValueError("trajectory_history_size must not exceed M")
        self.trajectory_scorer = (TrajectoryHistoryScorer(
            self.D, self.input_size, history_size,
            hidden_dim=int(getattr(args, "trajectory_history_dim", 32)),
            max_bias=float(getattr(args, "trajectory_history_max_bias", 0.5)),
        ) if history_size else None)
        self.trajectory_history_stages = getattr(args, "trajectory_history_stages", "both")
        if self.trajectory_history_stages not in {"both", "c1", "c2", "none"}:
            raise ValueError("trajectory_history_stages must be both, c1, c2, or none")
        # === === ===

        # === Time Positional Embedding for memory ===
        self.t_embedding = nn.Parameter(torch.zeros(1, self.M + 1, self.D))
        nn.init.trunc_normal_(self.t_embedding, std=0.02)
        # === === ===

        self.candidate_scorer = None
        self.memory_reliability = None
        if getattr(args, "history_candidate_scoring", False):
            if self.trajectory_scorer is not None:
                raise ValueError("history_candidate_scoring replaces the optional trajectory scorer")
            self.candidate_scorer = CandidateHistoryScorer(
                self.D, self.input_size,
                hidden_dim=int(getattr(args, "candidate_score_dim", 64)),
                num_layers=int(getattr(args, "candidate_score_layers", 2)), num_heads=nhead)
        if getattr(args, "memory_reliability_weighting", False) or self.candidate_scorer is not None:
            self.memory_reliability = nn.Linear(2, nhead)
            nn.init.zeros_(self.memory_reliability.weight)
            nn.init.zeros_(self.memory_reliability.bias)
        self.uses_history = (self.trajectory_scorer is not None or self.candidate_scorer is not None
                             or self.memory_reliability is not None)
        rgb_size = getattr(args, "rgb_memory_size", 0)
        if type(rgb_size) is not int or rgb_size < 0:
            raise ValueError("rgb_memory_size must be a nonnegative integer")
        self.rgb_memory_reader = (RGBMemoryReader(self.memory_attention, self.t_embedding, rgb_size, self.D)
                                  if rgb_size else None)

    @property
    def rgb_memory_size(self):
        return self.rgb_memory_reader.size if self.rgb_memory_reader is not None else 0

    @property
    def pretrained_memory_prefixes(self):
        return (("memory_reliability.",) if self.memory_reliability is not None else ()) + (
            ("rgb_memory_reader.",) if self.rgb_memory_reader is not None else ())

    def initialize_missing_memory(self, missing):
        """Only whole new components may be absent from an older checkpoint."""
        keys = set(self.state_dict())
        for prefix in self.pretrained_memory_prefixes:
            expected = {key for key in keys if key.startswith(prefix)}
            absent = expected.intersection(missing)
            if absent and absent != expected:
                raise RuntimeError(f"Incomplete memory component in checkpoint: {prefix}")
            if absent and prefix == "rgb_memory_reader.":
                self.rgb_memory_reader.initialize_from(self.memory_attention, self.t_embedding)

    def _decode_queries(self, query, features, bank, rgb_bank=None, p_mask=0):
        """Share causal decoding between training, online and streamed inference."""
        memory = bank.values
        bias = self._memory_bias(bank.history)
        rgb_memory = None if rgb_bank is None else rgb_bank.values
        rgb_bias = None if rgb_bank is None else self._memory_bias(rgb_bank.history)
        for index in range(self.decoder_layer_num):
            query = self.feature_attention[index](query, features, features)
            query = self.query_attention[index](query, query, query)
            recent, memory = read_memory(self.memory_attention[index], query, memory, bank.mask,
                                         self.t_embedding, bias, p_mask)
            if rgb_bank is not None:
                rgb, rgb_memory = self.rgb_memory_reader.read(
                    index, query, rgb_memory, rgb_bank.mask, rgb_bias, p_mask)
                query = self.rgb_memory_reader.fuse(index, recent, rgb, rgb_bank.mask)
            else:
                query = recent
        return self.projection1(query)

    def _memory_bias(self, history):
        if self.memory_reliability is None:
            return None
        if history is None:
            raise ValueError("Memory weighting requires synchronized xy/v/u history")
        bias = self.memory_reliability(history[..., 2:].detach()).transpose(1, 2)
        return F.pad(bias, (0, 1))  # The current query has no reliability bias.

    def _predict_position(self, q, features, patches, rerank_scores, memory, history, mask):
        """Share localization between training, online and streaming inference."""
        if self.candidate_scorer is None:
            c2 = self.multiscale_correlation(q, *features, trajectory_history=history,
                                            temporal_mask=mask, stage=1)
            patch = indices_to_coords(c2.detach().argmax(-1).unsqueeze(1),
                                     self.input_size, self.stride).squeeze(1)
            offsets, visibility, uncertainty = self.prediction_head(q, *features, patch)
            return patch, offsets, visibility, uncertainty, c2, None

        offsets, visibility, uncertainty = self.prediction_head(q, *features, patches)
        positions = patches[0] + offsets[-1]
        scores = self.candidate_scorer(positions, features, memory, history, mask, self.t_embedding)
        # Initialization uses the already supervised reranking head, not empty tokens.
        scores = torch.where(mask.all(-1, keepdim=True), rerank_scores[0], scores)
        best = scores.detach().argmax(-1)
        points = torch.arange(q.shape[1], device=q.device)
        patch = patches[:, points, best]
        selected_offsets = offsets[:, points, best]
        candidates = {"O_candidates": offsets, "S_candidates": scores,
                      "P_candidates": positions}
        return patch, selected_offsets, visibility, uncertainty, None, candidates

    def _backbone_features(self, video, events=None, img_ifnew=None, frontend_state=None):
        """Keep event arguments out of the RGB backbone's interface."""
        if isinstance(self.backbone, EventBackbone):
            return self.backbone(video, events=events, img_ifnew=img_ifnew, frontend_state=frontend_state)
        if events is not None:
            raise ValueError("events were provided, but event_channels is not configured")
        return self.backbone(video)

    def _make_transformer_layer(self, layer_num, nhead=4):
        layers = []
        for _ in range(layer_num):
            layer = MHA_Block(self.D, nhead)
            layers.append(layer)
        return nn.ModuleList(layers)


    def memory_extension(self, new_memory_size):
        if self.trajectory_scorer is not None and new_memory_size < self.trajectory_scorer.history_size:
            raise ValueError("Inference memory must contain at least trajectory_history_size slots")
        if type(new_memory_size) is not int or new_memory_size < 1:
            raise ValueError("Memory size must be a positive integer")
        self.t_embedding = nn.Parameter(resize_time_embedding(self.t_embedding.detach(), new_memory_size))

        self.M = new_memory_size
        
    def multiscale_correlation(self, q, h4, h8, h16, h32,
                              trajectory_history=None, temporal_mask=None, stage=0):
        # :args q: (1, N, D)
        # :args h4: (1, P, D)
        # :args f8: (1, P // 4, D)
        # :args f16: (1, P // 16, D)
        # :args f32: (1, P // 32, D)

        N = q.shape[1]                        # Number of queries
        _, P, D = h4.shape                    # Number of tokens

        q_normalized = F.normalize(q, p=2, dim=-1)    # (1, N, D)

        c4 = torch.einsum("bnd,bpd->bnp",  q_normalized, F.normalize(h4, p=2, dim=-1))    # (1, N, P)
        c8 = torch.einsum("bnd,bpd->bnp",  q_normalized, F.normalize(h8, p=2, dim=-1))    # (1, N, P // 4)
        c16 = torch.einsum("bnd,bpd->bnp",  q_normalized, F.normalize(h16, p=2, dim=-1))  # (1, N, P // 16)
        c32 = torch.einsum("bnd,bpd->bnp",  q_normalized, F.normalize(h32, p=2, dim=-1))  # (1, N, P // 32)

        c4 = c4.view(N, self.Hf, self.Wf)                      # (N, H4, W4)
        c8 = c8.view(N, self.Hf // 2, self.Wf // 2)            # (N, H8, W8)
        c16 = c16.view(N, self.Hf // 4, self.Wf // 4)          # (N, H16, W16)
        c32 = c32.view(N, self.Hf // 8, self.Wf // 8)          # (N, H32, W32)

        c4 = c4.unsqueeze(1) # (N, 1, H4, W4)
        c8 = F.interpolate(c8.unsqueeze(1), size=(self.Hf, self.Wf), mode="bilinear", align_corners=False)   # (N, 1, H4, W4)
        c16 = F.interpolate(c16.unsqueeze(1), size=(self.Hf, self.Wf), mode="bilinear", align_corners=False) # (N, 1, H4, W4)
        c32 = F.interpolate(c32.unsqueeze(1), size=(self.Hf, self.Wf), mode="bilinear", align_corners=False) # (N, 1, H4, W4)

        c = torch.cat([c4, c8, c16, c32], dim=1)     # (N, 4, H4, W)
        c = self.ms_corr_proj(c)                     # (N, 1, H4, W)

        c = c.view(1, N, self.P)                     # (N, P)

        if (self.trajectory_scorer is not None and trajectory_history is not None
                and self.trajectory_history_stages in {"both", f"c{stage + 1}"}):
            c = self.trajectory_scorer(c, q, trajectory_history, temporal_mask, stage)
        return c
    
    # Multiscale ViT-Adapter
    def forward(self, video, queries, p_mask=0.1, events=None, img_ifnew=None,
                warmup_steps=0, _state=None):
        if warmup_steps:
            if not 0 < warmup_steps < video.shape[1]:
                raise ValueError("warmup_steps must leave a nonempty supervised segment")
            state = {}
            with torch.no_grad():
                self.forward(video[:, :warmup_steps], queries.clone(), p_mask=p_mask,
                             events=None if events is None else events[:, :warmup_steps],
                             img_ifnew=None if img_ifnew is None else img_ifnew[:, :warmup_steps],
                             _state=state)
            # AMP must not reuse weight casts cached by the no-grad prefix.
            torch.clear_autocast_cache()
            queries = queries.clone()
            queries[..., 0] -= warmup_steps
            return self.forward(video[:, warmup_steps:], queries, p_mask=p_mask,
                                events=None if events is None else events[:, warmup_steps:],
                                img_ifnew=None if img_ifnew is None else img_ifnew[:, warmup_steps:],
                                _state=state)
        # :args video: (B, T, 3, H_in, W_in) in range [0, 255]
        # :args gt_tracks: (B, T, N, 2) in range [0, H_in] and [0, W_in]
        # :args queries: (B, N, 3) where 3 is (t, x, y) in range [0, 255]
        B, T, _, H_in, W_in = video.shape
        if self.rgb_memory_reader is not None and events is not None and img_ifnew is None:
            raise ValueError("Dual FE memory requires explicit img_ifnew")
        rgb_arrivals = (torch.ones(B, T, device=video.device, dtype=torch.bool) if img_ifnew is None
                        else torch.as_tensor(img_ifnew, device=video.device, dtype=torch.bool))
        if rgb_arrivals.shape != (B, T):
            raise ValueError("img_ifnew must have shape (B, T)")
        M = self.M
        D = self.D
        N = queries.shape[1]                        # Number of queries
        K = self.K
        layer_num_pred_head = self.prediction_head.layer_num
        device = video.device

        # === Scaling ====
        # scale to [0, 1] range
        queries[..., 2] = (queries[..., 2] / H_in)
        queries[..., 1] = (queries[..., 1] / W_in)

        video = video.view(B * T, 3, H_in, W_in) / 255.0                        # (B * T, 3, H, W), to [0, 1]
        video = F.interpolate(video, size=self.input_size, mode="bilinear", align_corners=False)
        video = video.view(B, T, 3, self.H, self.W)                             # (B, T, 3, H, W)
        if events is not None:
            if events.shape[:2] != (B, T) or events.shape[-2:] != (H_in, W_in):
                raise ValueError("events must match the video batch, time, height, and width")
            event_channels = events.shape[2]
            events = F.interpolate(
                events.reshape(B * T, event_channels, H_in, W_in),
                size=self.input_size, mode="bilinear", align_corners=False,
            ).reshape(B, T, event_channels, self.H, self.W)
        # === === ===

        # == Feature Projection ==
        frontend_state = None
        if _state is not None and isinstance(self.backbone, EventBackbone):
            if "frontend" not in _state:
                _state["frontend"] = self.backbone.create_state()
            frontend_state = _state["frontend"]
        f4, f8, f16, f32 = self._backbone_features(
            video, events=events, img_ifnew=img_ifnew, frontend_state=frontend_state)
        f_fused = self.fpn(f4, f8, f16, f32)                              # (B, T, P, D)
        # === === ===

        # === Sample query features from f_fused ===
        query_init = torch.zeros(B, N, D, device=device)            # empty tensor of shape B, N, D
        if _state is not None and "query_init" in _state:
            query_init = _state["query_init"].clone()
        query_times = queries[:, :, 0].long()                       # (B, N)
        query_coords_norm = queries[..., 1:].clone()                # (B, N, 2)
        query_coords_norm  = query_coords_norm * 2 - 1              # (B, N, 2), in range [-1, 1]
        for b in range(B):

            for t in range(T):
                mask = (query_times[b] == t)        # (N,)
                if not mask.any():
                    continue

                f_b_t = f_fused[b, t].view(1, self.Hf, self.Wf, self.D)                     # (1, H4, W4, D)
                f_b_t = f_b_t.permute(0, 3, 1, 2)                                           # (1, D, H4, W4)
                idxs = mask.nonzero(as_tuple=True)[0]                           # (N_t,)
                pts = query_coords_norm[b, idxs].unsqueeze(0).unsqueeze(2)                                         # (1, N_t, 1, 2)

                # Sample Points
                centers = F.grid_sample(f_b_t, pts, mode='bilinear', padding_mode='border', align_corners=False)   # (1, D, N_t, 1)
                centers = centers.squeeze(-1).squeeze(0).permute(1, 0)                                             # (N_t, D)
                query_init[b, idxs] = centers                                                                      # (N_t, D)
            # === === ===

        memories = [PointMemory.empty(query_init, N, M, D, self.uses_history) for _ in range(B)]
        rgb_memories = ([PointMemory.empty(query_init, N, self.rgb_memory_size, D, self.uses_history)
                         for _ in range(B)] if self.rgb_memory_reader is not None else None)
        if _state is not None and "memories" in _state:
            memories, rgb_memories = _state["memories"], _state["rgb_memories"]

        # === Output initialization ===
        C1 = torch.zeros(B, T, N, self.P, device=device)                        # (B, T, N, P)
        C2 = torch.zeros(B, T, N, self.P, device=device) if self.candidate_scorer is None else None
        O = torch.zeros(B, T, layer_num_pred_head, N, 2, device=device)         # (B, layer_num_pred_head, T, N, 2)

        V_logit = torch.zeros(B, T, N, device=device)      # (B, T, N)
        U_logit = torch.zeros(B, T, N, device=device)      # (B, T, N)
        
        P_patch = torch.zeros(B, T, N, 2, device=device)                        # (B, T, N, 2)

        U_logit_top_k = torch.zeros(B, T, N, K, device=device)                  # (B, T, N, K)
        S_logit_top_k = torch.zeros(B, T, N, K, device=device)                  # (B, T, N, K)
        P_patch_top_k = torch.zeros(B, T, N, K, 2, device=device)               # (B, T, N, K, 2)
        candidate_outputs = {}
        if self.candidate_scorer is not None:
            candidate_outputs = {
                "O_candidates": torch.zeros(B, T, layer_num_pred_head, N, K, 2, device=device),
                "S_candidates": torch.zeros(B, T, N, K, device=device),
                "P_candidates": torch.zeros(B, T, N, K, 2, device=device),
            }
        # === === ===

        for t in range(T):
            for b in range(B):
                available_queries = (query_times[b] <= t)                        # (N,)
            
                N_t = available_queries.sum().item()                             # Number of available queries in this batch
                if N_t == 0:
                    print(f"Warning: No available queries at time {t} for batch {b}. Skipping.")
                    continue
                
                bank = memories[b].select(available_queries)
                memory_mask_t, history_t = bank.mask, bank.history
                rgb_bank = None if rgb_memories is None else rgb_memories[b].select(available_queries)
                
                q_t = query_init[b, available_queries].clone().unsqueeze(0)      # (1, N_t, D)

                f_fused_t = f_fused[b, t].unsqueeze(0)                           # (1, P, D)
                f4_t = f4[b, t].unsqueeze(0)                                     # (1, P, D)
                f8_t = f8[b, t].unsqueeze(0)                                     # (1, P // 4, D)
                f16_t = f16[b, t].unsqueeze(0)                                   # (1, P // 16, D)
                f32_t = f32[b, t].unsqueeze(0)                                   # (1, P // 64, D)

                q_t = self._decode_queries(q_t, f_fused_t, bank, rgb_bank, p_mask=p_mask)
                # === === ===

                # === Re-ranking ===
                c1 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t,
                                                 trajectory_history=history_t, temporal_mask=memory_mask_t, stage=0)

                q_t, p_patch_top_k, u_logit_top_k, s_logit_top_k = self.reranking_head(q_t, f4_t, f8_t, f16_t, f32_t, c1)  # (1, N_t, D), (1, N_t, K, 2), (1, N_t, K), (1, N_t, K)
                q_t = self.projection2(q_t)                                                             # (1, N_t, D)
                p_patch, o, v_logit, u_logit, c2, candidates = self._predict_position(
                    q_t, (f4_t, f8_t, f16_t, f32_t), p_patch_top_k, s_logit_top_k,
                    bank.values, history_t, memory_mask_t)
                # === === ===

                C1[b, t, available_queries] = c1.squeeze(0).float()         # (1, N_t, P) 
                if C2 is not None:
                    C2[b, t, available_queries] = c2.squeeze(0).float()
                if candidates is not None:
                    candidate_outputs["O_candidates"][b, t, :, available_queries] = candidates["O_candidates"].float()
                    candidate_outputs["S_candidates"][b, t, available_queries] = candidates["S_candidates"].float()
                    candidate_outputs["P_candidates"][b, t, available_queries] = candidates["P_candidates"].float()
                O[b, t, :, available_queries] = o.float()                                        # (N, 2)
                V_logit[b, t, available_queries] = v_logit.float()                # (N)
                U_logit[b, t, available_queries] = u_logit.float()                # (N)
                P_patch[b, t, available_queries] = p_patch.squeeze(1).float()                 # (N, 2)

                U_logit_top_k[b, t, available_queries] = u_logit_top_k.squeeze(0).float()      # (N, K)
                S_logit_top_k[b, t, available_queries] = s_logit_top_k.squeeze(0).float()      # (N, K)
                P_patch_top_k[b, t, available_queries] = p_patch_top_k.squeeze(0).float()      # (N, K, 2)

                entry = (trajectory_entry(p_patch[0] + o[-1], v_logit, u_logit, self.input_size)
                         if self.uses_history else None)
                memories[b] = memories[b].append(q_t[0], entry, available_queries)
                if rgb_memories is not None and rgb_arrivals[b, t]:
                    rgb_memories[b] = rgb_memories[b].append(q_t[0], entry, available_queries)

            
        # P = indices_to_coords(torch.argmax(C1.detach(), dim=-1), self.input_size, self.stride)   # (B, T, N, 2), in range [H, W]
        P = P_patch + O[:, :, -1]
        P[..., 0] = (P[..., 0] / self.W) * W_in
        P[..., 1] = (P[..., 1] / self.H) * H_in

        # === Output ===
        # disctionary
        out = {"P": P,
                "C1": C1, "O": O,
                "V_logit": V_logit, "U_logit": U_logit, 
                "P_patch": P_patch, 
                "U_logit_top_k": U_logit_top_k, "S_logit_top_k": S_logit_top_k, "P_patch_top_k": P_patch_top_k}
        if C2 is not None:
            out["C2"] = C2
        out.update(candidate_outputs)
        if _state is not None:
            _state.update(query_init=query_init, memories=memories, rgb_memories=rgb_memories)

        return out
    

    def forward_online(self, video, queries):

        B, T, _, H_in, W_in = video.shape
        M = self.M
        D = self.D
        N = queries.shape[1]                        # Number of queries
        K = self.K
        device = video.device

        # === Scaling ====
        # scale to [0, 1] range
        queries[..., 2] = (queries[..., 2] / H_in)
        queries[..., 1] = (queries[..., 1] / W_in)

        video = video.view(B * T, 3, H_in, W_in) / 255.0                        # (B * T, 3, H, W), to [0, 1]
        video = F.interpolate(video, size=self.input_size, mode="bilinear", align_corners=False)
        video = video.view(B, T, 3, self.H, self.W)                             # (B, T, 3, H, W)
        # === === ===

        # === Sample query features from f_fused ===
        query_init = torch.zeros(B, N, D, device=device)            # empty tensor of shape B, N, D
        query_times = queries[:, :, 0].long()                       # (B, N)
        query_coords_norm = queries[..., 1:].clone()                # (B, N, 2)
        query_coords_norm  = query_coords_norm * 2 - 1              # (B, N, 2), in range [-1, 1]

        memories = [PointMemory.empty(query_init, N, M, D, self.uses_history) for _ in range(B)]
        rgb_memories = ([PointMemory.empty(query_init, N, self.rgb_memory_size, D, self.uses_history)
                         for _ in range(B)] if self.rgb_memory_reader is not None else None)

        # === Output initialization ===
        V_logit = torch.zeros(B, T, N, device=device)      # (B, T, N)
        U_logit = torch.zeros(B, T, N, device=device)      # (B, T, N)
        P = torch.zeros(B, T, N, 2, device=device)                        # (B, T, N, 2)
        # === === ===

        for t in range(T):
            for b in range(B):

                available_queries = (query_times[b] <= t)                        # (N,)

                N_t = available_queries.sum().item()                             # Number of available queries in this batch
                if N_t == 0:
                    continue
                
                bank = memories[b].select(available_queries)
                memory_mask_t, history_t = bank.mask, bank.history
                rgb_bank = None if rgb_memories is None else rgb_memories[b].select(available_queries)

                # === Visual Backbone Feedforward ===
                f4_t, f8_t, f16_t, f32_t = self._backbone_features(video[b, t].unsqueeze(0).unsqueeze(0))     # (1, 1, P, D), (1, 1, P // 4, D), (1, 1, P // 16, D), (1, 1, P // 64, D)
                f_fused_t = self.fpn(f4_t, f8_t, f16_t, f32_t)                                      # (1, 1, P, D)
                
                f4_t = f4_t.squeeze(0)                                                              # (1, P, D)
                f8_t = f8_t.squeeze(0)                                                              # (1, P // 4, D)
                f16_t = f16_t.squeeze(0)                                                            # (1, P // 16, D)
                f32_t = f32_t.squeeze(0)                                                            # (1, P // 64, D)
                f_fused_t = f_fused_t.squeeze(0)                                                    # (1, P, D)
                # === === ===

                # === Sample query features from f_fused ===
                queried_now = query_times[b] == t                                           # (N,)
                if queried_now.any():
                    f_b_t = f_fused_t.view(1, self.Hf, self.Wf, self.D)                     # (1, H4, W4, D)
                    f_b_t = f_b_t.permute(0, 3, 1, 2)                                       # (1, D, H4, W4)
                    idxs = queried_now.nonzero(as_tuple=True)[0]                            # (N_now_t,)
                    pts = query_coords_norm[b, idxs].unsqueeze(0).unsqueeze(2)                                         # (1, N_now_t, 1, 2)

                    # Sample Points
                    centers = F.grid_sample(f_b_t, pts, mode='bilinear', padding_mode='border', align_corners=False)   # (1, D, N_now_t, 1)
                    centers = centers.squeeze(-1).squeeze(0).permute(1, 0)                                             # (N_now_t, D)
                    query_init[b, idxs] = centers                                                                      # (N_now_t, D)
                # === === ===
                
                q_t = query_init[b, available_queries].clone().unsqueeze(0)      # (1, N_t, D)


                q_t = self._decode_queries(q_t, f_fused_t, bank, rgb_bank, p_mask=0)
                # === === ===

                # === Re-ranking ===
                c1 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t,
                                                 trajectory_history=history_t, temporal_mask=memory_mask_t, stage=0)
                q_t, p_patch_top_k, _, s_logit_top_k = self.reranking_head(q_t, f4_t, f8_t, f16_t, f32_t, c1)
                q_t = self.projection2(q_t)                                                            # (1, N_t, D)

                p_patch, o, v_logit, u_logit, _, _ = self._predict_position(
                    q_t, (f4_t, f8_t, f16_t, f32_t), p_patch_top_k, s_logit_top_k,
                    bank.values, history_t, memory_mask_t)
                # === === ===

                P[b, t, available_queries] = p_patch[0] + o[-1]                   # (1, N_t, 2)
                V_logit[b, t, available_queries] = v_logit.float()                # (N)
                U_logit[b, t, available_queries] = u_logit.float()                # (N)

                entry = (trajectory_entry(p_patch[0] + o[-1], v_logit, u_logit, self.input_size)
                         if self.uses_history else None)
                memories[b] = memories[b].append(q_t[0], entry, available_queries)
                if rgb_memories is not None:
                    rgb_memories[b] = rgb_memories[b].append(q_t[0], entry, available_queries)

        # === Output ===
        P[..., 0] = (P[..., 0] / self.W) * W_in
        P[..., 1] = (P[..., 1] / self.H) * H_in
        out = {"P": P, "V_logit": V_logit, "U_logit": U_logit}

        return out

    # ==== Functions for Streaming inference below ==== #
    @torch.no_grad()
    def extract_frame_features(self, frame, events=None, img_ifnew=None, frontend_state=None):
        """
        Extract multi-scale features from a single frame.
        
        :args frame: Tensor of shape (1, 3, H_in, W_in) in range [0, 255]
        :args events: Optional time surface of shape (1, C, H_in, W_in)
        :args img_ifnew: New RGB arrival flag for motion fusion.
        :args frontend_state: Caller-owned motion state for streaming inference.
        :return: Tuple of (f4_t, f8_t, f16_t, f32_t, f_fused_t) where all are (1, P_x, D)
        """
        if events is not None:
            if events.ndim != 4 or events.shape[0] != frame.shape[0] or events.shape[-2:] != frame.shape[-2:]:
                raise ValueError("events must match the frame batch, height, and width")
            events = F.interpolate(events, size=self.input_size, mode="bilinear", align_corners=False)
            events = events.unsqueeze(1)
        # === Scaling ====
        frame = frame / 255.0                               # to [0, 1]
        frame = F.interpolate(frame, 
                              size=self.input_size, 
                              mode="bilinear", 
                              align_corners=False)          # (1, 3, H, W)
        frame = frame.unsqueeze(1)                          # (1, 1, 3, H, W)
        # === === ===
        
        # === Visual Backbone Feedforward ===
        flags = None if img_ifnew is None else torch.as_tensor(img_ifnew, device=frame.device).reshape(1, 1)
        f4_t, f8_t, f16_t, f32_t = self._backbone_features(
            frame, events=events, img_ifnew=flags, frontend_state=frontend_state,
        )
        f_fused_t = self.fpn(f4_t, f8_t, f16_t, f32_t)      # (1, 1, P, D)
        
        # Squeeze out batch dimension
        f4_t = f4_t.squeeze(0)                              # (1, P, D)
        f8_t = f8_t.squeeze(0)                              # (1, P // 4, D)
        f16_t = f16_t.squeeze(0)                            # (1, P // 16, D)
        f32_t = f32_t.squeeze(0)                            # (1, P // 64, D)
        f_fused_t = f_fused_t.squeeze(0)                    # (1, P, D)
        # === === ===
        
        return f4_t, f8_t, f16_t, f32_t, f_fused_t

    @torch.no_grad()
    def track_frame(self, q_init, temporal_mask, point_memory, frame_features, H_in, W_in,
                    trajectory_history=None, return_uncertainty=False, rgb_memory=None):
        """
        Track queries through one frame using pre-extracted features.
        
        :args q_init: (N, D) - Query features
        :args temporal_mask: (N, M) - Temporal mask (True if masked)
        :args point_memory: (N, M, D) - Point memory
        :args frame_features: Tuple of (f4_t, f8_t, f16_t, f32_t, f_fused_t)
        :args H_in: Original frame height
        :args W_in: Original frame width
        :args trajectory_history: Optional (N, M, 4), normalized xy and past v/u probabilities
        :args return_uncertainty: Include uncertainty logits in the returned tuple
        :return: (p, v_logit, q_new) where p is (N, 2), v_logit is (N,), q_new is (N, D)
        """
        f4_t, f8_t, f16_t, f32_t, f_fused_t = frame_features

        if self.rgb_memory_reader is not None and rgb_memory is None:
            raise ValueError("Dual memory requires the point's RGB history")
        bank = PointMemory(point_memory, temporal_mask, trajectory_history)
        memory_mask_t, history_t = bank.mask, bank.history
        q_t = self._decode_queries(q_init.unsqueeze(0), f_fused_t, bank, rgb_memory)
        # === === ===

        # === Re-ranking ===
        c1 = self.multiscale_correlation(q_t, f4_t, f8_t, f16_t, f32_t,
                                         trajectory_history=history_t, temporal_mask=memory_mask_t, stage=0)
        q_t, p_patch_top_k, _, s_logit_top_k = self.reranking_head(q_t, f4_t, f8_t, f16_t, f32_t, c1)
        q_t = self.projection2(q_t)                                                            # (1, N, D)

        p_patch, o, v_logit, u_logit, _, _ = self._predict_position(
            q_t, (f4_t, f8_t, f16_t, f32_t), p_patch_top_k, s_logit_top_k,
            point_memory, history_t, memory_mask_t)
        # === === ===

        # === Final predictions (point_mechanism='a') ===
        p = p_patch[0] + o[-1]                   # (N, 2)
        
        # Scale back to original image coordinates
        p[..., 0] = (p[..., 0] / self.W) * W_in
        p[..., 1] = (p[..., 1] / self.H) * H_in
        # === === ===

        if return_uncertainty:
            return p, v_logit, q_t.squeeze(0), u_logit
        return p, v_logit, q_t.squeeze(0)  # (N, 2), (N), (N, D)
