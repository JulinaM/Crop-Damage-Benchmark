# Change detection configs -- TODO

Mirrors `configs/segmentation/`'s layout once implemented:

```
change_detection/
  terramind_flood.yaml   terramind_burnt.yaml   terramind_pooled.yaml
  unet_flood.yaml        unet_burnt.yaml        unet_pooled.yaml
  prithvi_flood.yaml     prithvi_burnt.yaml     prithvi_pooled.yaml
```

Not implemented yet. Per CLAUDE.md sec.1, Task B (change detection) predicts
the pre->post change mask directly, as opposed to Task A (segmentation)'s
per-pixel damage class on post-event imagery. Note: `Trainer._forward()`
currently always runs the Siamese before/after + change_fusion path
regardless of task, so before writing these configs, decide whether Task B
here means a different *label target* through that same architecture, or
needs its own forward path -- see the "Task B isn't actually implemented as
a separate path" discussion from the experiment-design review.
