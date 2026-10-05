# Optional frame-event motion frontends

Existing configs default to `event_fusion_type: simple`. The original fusion,
checkpoint keys, tracker heads, and point memory keep their existing behavior.

`model/backbone.py` contains the RGB-only `Backbone`. The event extension lives
in `model/event_backbone.py` as `EventBackbone(Backbone)`, selected by
`event_channels` when TrackOn is constructed. Both return the same four token
scales. TrackOn routes batch and streaming feature calls through
`_backbone_features`; Predictor asks the backbone to `create_state()` at reset.
Inherited parameters retain their original `backbone.*` checkpoint paths, and
event parameters remain under `backbone.event_fusion.*` for event-only training.

Use `config/train_frame_event_motion.yaml` for either new frontend:

| Option | History |
| --- | --- |
| `motion_pool` | Independently encoded event motion maps since the latest RGB; local learned sampling and per-pixel attention select history. |
| `motion_state` | One updated motion map plus the previous anchor-relative warp; local sampling and gates update the state. |

`event_motion_dim` controls the stride-8 motion channels. `event_history_size`
bounds the pool; the single-state variant always keeps one entry. No time
embedding is used. Both use the existing current-event encoder and residual
gates, predict a two-channel target-to-anchor sampling field, warp the original
RGB content at four scales, then add target-grid positional embeddings.
The event `encoder` is an ordinary module returning stride-4, `D`-channel maps;
it can be replaced by an adapted event backbone without changing the tracker.

## Event encoder

Existing configs keep the original two-convolution encoder and its checkpoint
keys. To use ETAP in either simple fusion or a motion frontend, add only:

```yaml
event_encoder: etap
event_pretrained_path: checkpoints/ETAP_v1_cvpr25.pth
```

Omit `event_pretrained_path` (or set it to `null`) for random initialization.
Download the [official checkpoint](https://drive.google.com/file/d/18mnwu8CsrVJvDXeRtvU0Wgp6shdxn0HD/view)
from [ETAP (CVPR 2025)](https://github.com/tub-rip/ETAP).
No network access occurs during model construction. The loader strictly loads
the checkpoint's `fnet.*` parameters, retaining its native 128 output channels.
A new trainable 1x1 projection maps them to `D`; the complete encoder returns
`(B, D, H/4, W/4)`. Raw fnet state dictionaries and `model`/`state_dict` wrappers
are accepted, including a `module.` prefix. The full TrackOn checkpoint remains
responsible for restoring the projection and fusion/motion modules.

The official input has 10 mixed-density event-stack channels. Our time surfaces
have the same channel count but different semantics; pretrained weights are an
initialization for fine-tuning, not a matching input distribution. Input channels
must match the checkpoint; no weights are silently skipped or reshaped.
The existing `train_event_only` filter includes both fnet and projection because
they remain under `backbone.event_fusion.encoder.*`. Changing encoders requires
a new training run; old simple-encoder checkpoints are not ETAP checkpoints.

Future encoders can be added to `build_event_encoder` in `model/event_encoder.py`,
or passed as an `encoder` module to `EventFeatureFusion`/`MotionEventFusion`.
Adapters must provide the same stride-4, `D`-channel output and process each event
step independently; temporal state continues to belong to the motion frontend.
The ETAP adaptation preserves source attribution and the license in
`licenses/ETAP.txt`.

Motion fields use feature-pixel units; resizing scales x/y displacements as well
as the grid. History-reader offsets are separate target-to-history sampling
fields. The motion head starts at zero; the event residual gates start at zero.
An entirely empty event group has zero warp and no event residual.

## Interfaces

Batch forward accepts `img_ifnew` of shape `(B, T)` alongside `events`.
Motion clips must start at an actual RGB anchor. `main_frame_event.py` selects
anchor-aligned clips only for the new options; existing sampling is unchanged.

Streaming calls provide the arrival flag explicitly:

```python
predictor.forward_frame(frame, events=events, img_ifnew=is_new_rgb)
```

`Predictor.frontend_state` owns the RGB anchor, motion history and latest field.
Each new RGB invalidates the old anchor's history; `Predictor.reset()` starts a
new sequence. Neither operation resets point memory on an ordinary RGB arrival.
The RGB backbone encodes new anchors only. Batch calls create their own state.

History is read before the current entry is appended, so future entries are
unavailable by construction. Pool attention additionally masks invalid entries
at per-sample RGB boundaries and includes a null entry for empty history. The
single-state reader masks its previous entry at the same boundary. This avoids
an unnecessary full triangular attention matrix in sequential execution.

Only frontend inference smoke checks are required for this change. No training
or dataset benchmark is needed to check shapes, reset, causality and streaming.
The new modes require their own trained weights before meaningful evaluation.
