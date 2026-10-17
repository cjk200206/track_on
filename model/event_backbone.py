"""Event extensions of the RGB TrackOn backbone."""

import torch

from model.backbone import Backbone
from model.event_encoder import build_event_encoder
from model.event_fusion import EventFeatureFusion
from model.event_motion import MotionEventFusion, MotionState
from model.event_recurrent import RecurrentEventFusion, RecurrentState


FUSION_TYPES = {"simple", "motion_pool", "motion_state", "recurrent", "recurrent_warp"}


def build_backbone(args):
    """Select the RGB or event backbone without changing checkpoint key paths."""
    fusion_type = getattr(args, "event_fusion_type", "simple")
    if fusion_type not in FUSION_TYPES:
        raise ValueError(f"Unknown event_fusion_type: {fusion_type}")
    if getattr(args, "event_channels", 0):
        return EventBackbone(args)
    if fusion_type != "simple":
        raise ValueError("Stateful fusion requires event_channels")
    return Backbone(args)


class EventBackbone(Backbone):
    """Simple residuals, causal RGB warping, or recurrent content fusion."""

    def __init__(self, args):
        event_channels = getattr(args, "event_channels", 0)
        if event_channels < 1:
            raise ValueError("EventBackbone requires positive event_channels")
        fusion_type = getattr(args, "event_fusion_type", "simple")
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"Unknown event_fusion_type: {fusion_type}")
        super().__init__(args)
        encoder = build_event_encoder(
            event_channels, self.D,
            name=getattr(args, "event_encoder", "simple"),
            pretrained_path=getattr(args, "event_pretrained_path", None),
        )
        self.stateful_frontend = fusion_type != "simple"
        self.recurrent_frontend = fusion_type in {"recurrent", "recurrent_warp"}
        if self.recurrent_frontend:
            self.event_fusion = RecurrentEventFusion(
                event_channels, self.D, encoder=encoder,
                with_warp=fusion_type == "recurrent_warp",
                motion_dim=getattr(args, "event_motion_dim", 64),
            )
        elif self.stateful_frontend:
            self.event_fusion = MotionEventFusion(
                event_channels, self.D, fusion_type,
                motion_dim=getattr(args, "event_motion_dim", 64),
                history_size=getattr(args, "event_history_size", 8),
                encoder=encoder,
            )
        else:
            self.event_fusion = EventFeatureFusion(event_channels, self.D, encoder=encoder)

    def create_state(self):
        if self.recurrent_frontend:
            return RecurrentState()
        return MotionState() if self.stateful_frontend else None

    def forward(self, video, events=None, img_ifnew=None, frontend_state=None):
        batch, time, _, height, width = video.shape
        assert (height, width) == (self.H, self.W), "Input video size must match input_size"
        if self.stateful_frontend:
            return self._stateful_features(video, events, img_ifnew, frontend_state)
        features = self._add_position(self._encode_rgb(video.reshape(batch * time, 3, height, width)))
        if events is not None:
            features = self.event_fusion(features, events)
        return self._to_tokens(features, batch, time)

    def _stateful_features(self, video, events, img_ifnew, frontend_state):
        """Encode each new RGB once; build current-grid features before adding PE."""
        batch, time, _, height, width = video.shape
        state = frontend_state if frontend_state is not None else self.create_state()
        if img_ifnew is None and events is not None:
            raise ValueError("Stateful fusion requires explicit img_ifnew flags")
        flags = (torch.ones(batch, time, device=video.device, dtype=torch.bool) if img_ifnew is None
                 else torch.as_tensor(img_ifnew, device=video.device, dtype=torch.bool))
        if time < 1:
            raise ValueError("Stateful fusion requires at least one time step")
        if flags.shape != (batch, time):
            raise ValueError("img_ifnew must have shape (B, T)")
        if state.anchor is None and not flags[:, 0].all():
            raise ValueError("Start a stateful clip at a new RGB anchor")
        if events is None:
            events = video.new_zeros(batch, time, self.event_fusion.event_channels, height, width)
        if events.shape != (batch, time, self.event_fusion.event_channels, height, width):
            raise ValueError("events must match the video and configured channels")

        indices = flags.reshape(-1).nonzero().flatten()
        anchors = state.anchor
        if indices.numel():
            images = video.reshape(batch * time, 3, height, width)[indices]
            anchors = self._encode_rgb(images)
        fresh = tuple(
            feat.new_zeros(batch * time, *feat.shape[1:]).index_copy(0, indices, feat)
            if indices.numel() else feat.new_zeros(batch * time, *feat.shape[1:])
            for feat in anchors
        )
        fresh = tuple(feat.reshape(batch, time, *feat.shape[1:]) for feat in fresh)
        encoded = self.event_fusion.encode_events(events.reshape(batch * time, *events.shape[2:]))
        encoded = tuple(feat.reshape(batch, time, *feat.shape[1:]) for feat in encoded)
        steps = []
        for t in range(time):
            steps.append(self.event_fusion.step(
                tuple(feat[:, t] for feat in fresh), tuple(feat[:, t] for feat in encoded), flags[:, t], state,
            ))

        features = tuple(
            torch.stack([step[scale] for step in steps], dim=1).flatten(0, 1)
            for scale in range(4)
        )
        return self._to_tokens(self._add_position(features), batch, time)
