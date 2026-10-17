"""Check causal content recurrence and reuse of the existing warp frontend."""

import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import Backbone
from model.event_backbone import EventBackbone
from model.event_motion import MotionState
from model.event_recurrent import RecurrentEventFusion, RecurrentState
from model.trackon import Track_On2
from model.trackon_predictor import Predictor


class TinyRGBEncoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.projection = nn.Conv2d(3, dim, 1)
        self.encoded_images = 0

    def forward(self, images):
        self.encoded_images += len(images)
        features = self.projection(images)
        return tuple(F.avg_pool2d(features, stride) for stride in (4, 8, 16, 32))


def init_tiny_backbone(backbone, args):
    nn.Module.__init__(backbone)
    backbone.D = args.D
    backbone.H, backbone.W = args.input_size
    backbone.vit_encoder = TinyRGBEncoder(args.D)
    backbone.pos_embedding = nn.Parameter(torch.randn(1, args.D, 8, 8))


class RecurrentFrontendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.video = torch.rand(2, 5, 3, 32, 32)
        self.events = torch.rand(2, 5, 10, 32, 32)
        self.flags = torch.tensor([[1, 0, 0, 1, 0], [1, 0, 1, 0, 0]], dtype=torch.bool)

    def backbone(self, mode):
        args = Namespace(D=16, input_size=(32, 32), event_channels=10,
                         event_fusion_type=mode, event_motion_dim=8)
        with patch.object(Backbone, "__init__", init_tiny_backbone):
            model = EventBackbone(args)
        model._encode_rgb = model.vit_encoder.forward
        return model.eval()

    def test_batch_matches_streaming_and_encodes_only_new_rgb(self):
        for mode in ("recurrent", "recurrent_warp", "motion_state", "motion_pool"):
            with self.subTest(mode=mode), torch.no_grad():
                model = self.backbone(mode)
                batch_state, stream_state = model.create_state(), model.create_state()
                batch = model(self.video, self.events, self.flags, batch_state)
                self.assertEqual(model.vit_encoder.encoded_images, int(self.flags.sum()))
                steps = [model(self.video[:, t:t + 1], self.events[:, t:t + 1],
                               self.flags[:, t:t + 1], stream_state) for t in range(5)]
                for scale, expected in enumerate(batch):
                    actual = torch.cat([step[scale] for step in steps], dim=1)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    self.assertEqual(expected.shape, (2, 5, (8 // 2 ** scale) ** 2, 16))
                if mode.startswith("recurrent"):
                    torch.testing.assert_close(batch_state.fused, stream_state.fused)

    def test_future_events_do_not_change_prefix(self):
        for mode in ("recurrent", "recurrent_warp"):
            with self.subTest(mode=mode), torch.no_grad():
                model = self.backbone(mode)
                expected = model(self.video, self.events, self.flags)
                changed = self.events.clone()
                changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
                actual = model(self.video, changed, self.flags)
                for before, after in zip(expected, actual):
                    torch.testing.assert_close(before[:, :3], after[:, :3], atol=0, rtol=0)

    def test_rgb_refresh_resets_only_the_corresponding_sample(self):
        for mode in ("recurrent", "recurrent_warp"):
            with self.subTest(mode=mode), torch.no_grad():
                model = self.backbone(mode)
                full = model(self.video, self.events, self.flags)
                # Sample 1 receives a fresh RGB at step 2, while sample 0 continues.
                fresh = model(self.video[1:2, 2:], self.events[1:2, 2:], self.flags[1:2, 2:])
                for expected, actual in zip(full, fresh):
                    torch.testing.assert_close(expected[1:2, 2:], actual, atol=2e-6, rtol=2e-5)

    def test_final_content_is_persisted_and_receives_temporal_gradients(self):
        for with_warp in (False, True):
            with self.subTest(with_warp=with_warp):
                fusion = RecurrentEventFusion(10, 16, with_warp=with_warp, motion_dim=8)
                state = RecurrentState()
                frames = tuple(torch.randn(1, 16, size, size) for size in (8, 4, 2, 1))
                first = torch.randn(1, 10, 32, 32, requires_grad=True)
                fusion.step(frames, fusion.encode_events(first), [True], state)
                last = fusion.step(frames, fusion.encode_events(torch.randn_like(first)), [False], state)
                self.assertIs(state.fused, last[-1])
                sum(feature.square().mean() for feature in last).backward()
                self.assertGreater(first.grad.abs().sum().item(), 0)
                for name, parameter in fusion.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if with_warp:
                    keys = fusion.state_dict()
                    self.assertTrue(any(key.startswith("encoder.") for key in keys))
                    self.assertFalse(any(key.startswith("motion.encoder.") for key in keys))
                    self.assertGreater(fusion.blend[-1].weight.grad.abs().sum().item(), 0)

    def test_hybrid_can_select_existing_warp_output(self):
        fusion = RecurrentEventFusion(10, 16, with_warp=True, motion_dim=8).eval()
        for gate in fusion.blend:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, 30)
        warp_state, hybrid_state = MotionState(), RecurrentState()
        frames = tuple(torch.randn(1, 16, size, size) for size in (8, 4, 2, 1))
        with torch.no_grad():
            for new_rgb in (True, False):
                encoded = fusion.encode_events(torch.randn(1, 10, 32, 32))
                expected = fusion.motion.step(frames, encoded, [new_rgb], warp_state)
                actual = fusion.step(frames, encoded, [new_rgb], hybrid_state)
                for before, after in zip(expected, actual):
                    torch.testing.assert_close(before, after)

    def test_state_requires_initial_rgb_and_complete_checkpoint_roundtrip(self):
        for mode in ("recurrent", "recurrent_warp"):
            with self.subTest(mode=mode):
                model = self.backbone(mode)
                with self.assertRaisesRegex(ValueError, "new RGB anchor"):
                    model(self.video[:, :1], self.events[:, :1], torch.zeros(2, 1, dtype=torch.bool))
                restored = self.backbone(mode)
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    for before, after in zip(model(self.video, self.events, self.flags),
                                             restored(self.video, self.events, self.flags)):
                        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_tracker_backward_and_predictor_reset_with_etap(self):
        for mode in ("recurrent", "recurrent_warp"):
            with self.subTest(mode=mode):
                args = Namespace(D=64, input_size=(32, 32), event_channels=10,
                                 event_fusion_type=mode, event_motion_dim=8, event_encoder="etap",
                                 M=3, M_i=3, K=4, decoder_layer_num=1,
                                 predicton_head_layer_num=1, rerank_layer_num=1, delta_v=0.8)
                with patch.object(Backbone, "__init__", init_tiny_backbone):
                    predictor = Predictor(args, support_grid_size=0)
                model = predictor.model
                model.backbone._encode_rgb = model.backbone.vit_encoder.forward
                video, events = self.video[:1, :3] * 255, self.events[:1, :3]
                flags = torch.tensor([[True, False, True]])
                queries = torch.tensor([[[0., 8., 8.], [0., 20., 20.]]])
                model.train()
                output = model(video, queries.clone(), p_mask=0, events=events, img_ifnew=flags)
                self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
                loss = sum(output[key].square().mean() for key in ("C1", "C2", "O", "V_logit", "U_logit"))
                loss.backward()
                gradient = model.backbone.event_fusion.encoder.fnet.conv1.weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
                predictor.eval()
                with torch.no_grad():
                    for t in range(3):
                        positions, visible = predictor.forward_frame(
                            video[:, t], new_queries=queries[0, :, 1:] if t == 0 else None,
                            events=events[:, t], img_ifnew=flags[:, t],
                        )
                        self.assertEqual(positions.shape, (2, 2))
                        self.assertEqual(visible.shape, (2,))
                        self.assertTrue(torch.isfinite(positions).all())
                self.assertEqual(predictor.t, 3)
                self.assertFalse(predictor.temporal_mask[:2].any())
                predictor.reset()
                self.assertIsNone(predictor.frontend_state.fused)
                self.assertIsNone(predictor.frontend_state.anchor)
                self.assertIsNone(predictor.point_memory)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
