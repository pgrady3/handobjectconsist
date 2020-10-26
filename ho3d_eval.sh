# Run her released model on test set
python evalho3dv2.py --resume releasemodels/ho3dv2/realonly/checkpoint_200.pth --val_split test --json_folder jsonres/res

# Run her released model on val set, should overfit since been trained on
#python evalho3dv2.py --resume releasemodels/ho3dv2/realonly/checkpoint_200.pth --val_split test --json_folder jsonres/res --split_mode=objects


# My model, running on real test split. Fair comparison against hers
# python evalho3dv2.py --resume checkpoints/ho3dv2_train_mini1/2020_10_01/debug/_frac1.0e+00lr5e-05_mom0.9_bs8__lmbeta5.0e-07_lmpr5.0e-05_lmrj3d5.0e-01_lovr3d5.0e-01seed0_fbn/checkpoint.pth --val_split test --json_folder jsonres/res