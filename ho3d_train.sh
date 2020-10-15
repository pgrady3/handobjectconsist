python trainmeshreg.py --freeze_batchnorm --workers 8 --train_datasets=ho3dv2 --val_dataset=ho3dv2 \
 --eval_freq 1 --val_split test --version 3 --mano_lambda_pose_reg=5e-5 --use_cache \
 --resume=checkpoints/ho3dv2_train_mini1/2020_09_30/debug/_frac1.0e+00lr5e-05_mom0.9_bs8__lmbeta5.0e-07_lmpr5.0e-05_lmrj3d5.0e-01_lovr3d5.0e-01seed0_fbn/checkpoint.pth