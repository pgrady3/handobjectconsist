# Run hers, but then we don't get ground truth hand annotations since test set doesn't have
#python run_all.py --resumes=releasemodels/ho3dv2/realonly/checkpoint_200.pth --split_mode=paper --split test

# Run my model on OBJECT split so we get GT hands
python run_all.py --split_mode=objects --split test \
 --resume checkpoints/ho3dv2_train_mini1/2020_10_01/debug/_frac1.0e+00lr5e-05_mom0.9_bs8__lmbeta5.0e-07_lmpr5.0e-05_lmrj3d5.0e-01_lovr3d5.0e-01seed0_fbn/checkpoint.pth






while true; do
    read -p "Do you want to copy generated pkl to other folder?" yn
    case $yn in
        [Yy]* ) cp all_samples.pkl ../align_hands/dataset/.; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
