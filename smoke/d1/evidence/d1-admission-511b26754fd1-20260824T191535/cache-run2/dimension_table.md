# D1 Interface Dimension Table

| Stage | Tensor shape | Dtype | Notes |
| --- | --- | --- | --- |
| Input | `B x 3 x 224 x 224` | FP32 | Resize, ImageNet normalize |
| DINOv2 block 3 | `B x 384 x 16 x 16` | FP16 cache | Patch size 14 |
| DINOv2 block 6 | `B x 384 x 16 x 16` | FP16 cache | Multi-layer feature |
| DINOv2 block 9 | `B x 384 x 16 x 16` | FP16 cache | Multi-layer feature |
| DINOv2 block 12 | `B x 384 x 16 x 16` | FP16 cache | Final dense feature |
| Global pooled | `B x 384` | FP16 cache | Final normalized CLS token |
| P3 adapter target | `B x 256 x 28 x 28` | FP32/AMP | Resize + channel projection required |
| P4 adapter target | `B x 512 x 14 x 14` | FP32/AMP | Resize + channel projection required |
| P5 adapter target | `B x 1024 x 7 x 7` | FP32/AMP | Resize + channel projection required |

The current 8.24 check validates the frozen feature/cache boundary. The P3/P4/P5 projections are the explicit next
implementation boundary for the D1 P0 train/predict route; this report does not claim that boundary is already wired.
