## Notes
* `ho3d_eval.sh` runs `run_all.py` and generates the pickle file needed
* "Paper" split uses full train/test set. Trainval 0:60k, val 60k:end, train 0:end. Test is full eval set
* "Object" split mode object makes all splits from training set. Train = trainval + val. Test is separate, but fully annotated
* I think strong/weak mode refers to her process of cutting out annotations. For my purposes, full is the way to go, as it is the sum of strong and weak. Test this, not sure


## Ideas to improve performance
* Don't use object split, use paper split `--split_mode paper`. 
* `--val_split val`
* `--scale_jittering 0.1`?
* `--max_rot 0.2`

* Loss weights are different

Must use `--version 2/3?`
Splits are very important. Using train/test, they're both actually splits from the train set

## Export predictions

Data structure shapes
```
Key BaseQueries.JOINTVIS. Shape torch.Size([4, 21])
Key BaseQueries.SIDE. Shape 2
Key BaseQueries.IMAGE. Shape torch.Size([4, 270, 480, 3])
Key TransQueries.AFFINETRANS. Shape torch.Size([4, 3, 3])
Key BaseQueries.JOINTS2D. Shape torch.Size([4, 21, 2])
Key TransQueries.JOINTS2D. Shape torch.Size([4, 21, 2])
Key BaseQueries.CAMINTR. Shape torch.Size([4, 3, 3])
Key TransQueries.CAMINTR. Shape torch.Size([4, 3, 3])
Key BaseQueries.OBJVERTS2D. Shape torch.Size([4, 18589, 2])
Key TransQueries.OBJVERTS2D. Shape torch.Size([4, 18589, 2])
Key BaseQueries.HANDVERTS2D. Shape torch.Size([4, 778, 2])
Key TransQueries.HANDVERTS2D. Shape torch.Size([4, 778, 2])
Key BaseQueries.JOINTS3D. Shape torch.Size([4, 21, 3])
Key TransQueries.JOINTS3D. Shape torch.Size([4, 21, 3])
Key BaseQueries.HANDVERTS3D. Shape torch.Size([4, 778, 3])
Key TransQueries.HANDVERTS3D. Shape torch.Size([4, 778, 3])
Key BaseQueries.OBJVERTS3D. Shape torch.Size([4, 18589, 3])
Key TransQueries.OBJVERTS3D. Shape torch.Size([4, 18589, 3])
Key BaseQueries.OBJFACES. Shape torch.Size([4, 34484, 3])
Key BaseQueries.OBJCANVERTS. Shape torch.Size([4, 18589, 3])
Key BaseQueries.OBJCANSCALE. Shape torch.Size([4])
Key BaseQueries.OBJCANTRANS. Shape torch.Size([4, 3])
Key TransQueries.CENTER3D. Shape torch.Size([4, 3])
Key TransQueries.IMAGE. Shape torch.Size([4, 3, 270, 480])
Key TransQueries.JITTERMASK. Shape torch.Size([4, 3, 270, 480])
Key dist2strong. Shape torch.Size([4])
Key dist2query. Shape torch.Size([4])

Network out verts3d: Shape torch.Size([4, 778, 3])
Network out joints3d: Shape torch.Size([4, 21, 3])
Network out shape: Shape torch.Size([4, 10])
Network out pose: Shape torch.Size([4, 18])
Network out joints2d: Shape torch.Size([4, 21, 2])
Network out recov_joints3d: Shape torch.Size([4, 21, 3])
Network out recov_handverts3d: Shape torch.Size([4, 778, 3])
Network out verts2d: Shape torch.Size([4, 778, 2])
Network out hand_pretrans: Shape torch.Size([4, 2])
Network out hand_prescale: Shape torch.Size([4, 1])
Network out hand_trans: Shape torch.Size([4, 1, 2])
Network out hand_scale: Shape torch.Size([4, 1, 1])
Network out obj_verts2d: Shape torch.Size([4, 18589, 2])
Network out obj_verts3d: Shape torch.Size([4, 18589, 3])
Network out recov_objverts3d: Shape torch.Size([4, 18589, 3])
Network out recov_objcorners3d: Shape None
Network out obj_scale: Shape torch.Size([4, 1, 1])
Network out obj_prescale: Shape torch.Size([4, 1])
Network out obj_prerot: Shape torch.Size([4, 3])
Network out obj_trans: Shape torch.Size([4, 1, 2])
Network out obj_pretrans: Shape torch.Size([4, 2])
Network out obj_corners2d: Shape None
Network out obj_corners3d: Shape None

```
