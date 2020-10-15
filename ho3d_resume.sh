python trainmeshreg.py --freeze_batchnorm --workers 8 --train_datasets=ho3dv2 --val_dataset=ho3dv2 \
 --eval_freq 1 --val_split test --version 3 --mano_lambda_pose_reg=5e-5 --use_cache --evaluate \
 --resume releasemodels/ho3dv2/realonly/checkpoint_200.pth --split_mode=paper