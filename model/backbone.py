import torch
import torch.nn as nn
import torch.nn.functional as F


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Backbone(nn.Module):
    """RGB backbone shared by the original tracker and event extensions."""

    def __init__(self, args):
        super().__init__()

        self.D = args.D 
        self.size = args.input_size
        self.stride = 4

        self.with_cp = args.grad_checkpoint
        self.vit_upsample_factor = args.vit_upsample_factor

        model_name = args.vit_backbone[:6]  # "dinov2" or "dinov3"
        arch_type = args.vit_backbone[7:]   # "s", "b", or "s_plus"

        if model_name == "dinov3":
            from model.vit_adapter.dinov3_adapter.dinov3_vit_adapter import ViTAdapter
            self.vit_encoder = ViTAdapter(deform_num_heads=12 if arch_type == "b" else 6,
                                          with_cp=self.with_cp,
                                          vit_upsample_factor=self.vit_upsample_factor,
                                          arch_type=arch_type,
                                          pretrained_model_path=getattr(args, "vit_pretrained_path", None))
            
        else:
            from model.vit_adapter.dinov2_adapter.dinov2_vit_adapter import ViTAdapter
            self.vit_encoder = ViTAdapter(deform_num_heads=12 if arch_type == "b" else 6,
                                          with_cp=self.with_cp,
                                          vit_upsample_factor=self.vit_upsample_factor,
                                          arch_type=arch_type)
        
        self.adapter_dim = self.vit_encoder.embed_dim

        # modulelist
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(self.adapter_dim, self.D)),
            nn.Sequential(nn.Linear(self.adapter_dim, self.D)),
            nn.Sequential(nn.Linear(self.adapter_dim, self.D)),
            nn.Sequential(nn.Linear(self.adapter_dim, self.D))
        ])

        # === Positional embedding ===

        self.H = self.size[0]
        self.W = self.size[1]
        self.Hf = self.H // self.stride
        self.Wf = self.W // self.stride
        self.P = int(self.Hf * self.Wf)                     # Number of tokens

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.D, self.Hf, self.Wf))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 3, 1, 1),
                persistent=False,
            )

    def project_feat(self, feat, idx):
        # :args feat: (B * T, self.adapter_dim, Hs, Ws)
        # :args scale: 1, 2, 4, 8

        BT, _, Hs, Ws = feat.shape

        feat = feat.permute(0, 2, 3, 1).reshape(BT, -1, self.adapter_dim)                                  # (B * T, Hs * Ws, 384)
        feat = self.projections[idx](feat)                                                    # (B * T, Hs * Ws, D)
        feat = feat.permute(0, 2, 1).reshape(BT, self.D, Hs, Ws)                              # (B * T, D, Hs, Ws)
        return feat

        
    def create_state(self):
        """Return optional caller-owned state for streaming inference."""
        return None

    def _encode_rgb(self, images):
        images = (images - self._resnet_mean) / self._resnet_std
        return tuple(self.project_feat(feat, i) for i, feat in enumerate(self.vit_encoder(images)))

    def _add_position(self, features):
        return tuple(
            feat + F.interpolate(self.pos_embedding, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            for feat in features
        )

    def _to_tokens(self, features, batch, time):
        return tuple(feat.flatten(2).transpose(1, 2).reshape(batch, time, -1, self.D) for feat in features)

    def forward(self, video):
        """Return four RGB token scales with shape (B, T, P_s, D)."""
        batch, time, _, height, width = video.shape
        assert (height, width) == (self.H, self.W), "Input video size must match input_size"
        features = self._encode_rgb(video.reshape(batch * time, 3, height, width))
        return self._to_tokens(self._add_position(features), batch, time)
