import argparse
from datetime import datetime
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import torch

from libyana.exputils.argutils import save_args
from libyana.modelutils import freeze

from meshreg.datasets import collate
from meshreg.netscripts import evalpass, reloadmodel, get_dataset
from tqdm import tqdm
from meshreg.datasets.queries import BaseQueries
from meshreg.models import manoutils
import pickle


plt.switch_backend("agg")


def to_cpu_npy(elem):
    if isinstance(elem, torch.Tensor):
        elem = elem.detach().cpu().numpy()

    return elem


def main(args):
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    # Initialize hosting
    dat_str = args.val_dataset
    now = datetime.now()
    exp_id = (
        f"checkpoints/{dat_str}_mini{args.mini_factor}/"
        f"{now.year}_{now.month:02d}_{now.day:02d}/"
        f"{args.com}_frac{args.fraction}_mode{args.mode}_bs{args.batch_size}_"
        f"objs{args.obj_scale_factor}_objt{args.obj_trans_factor}"
    )

    # Initialize local checkpoint folder
    save_args(args, exp_id, "opt")
    result_folder = os.path.join(exp_id, "results")
    os.makedirs(result_folder, exist_ok=True)
    pyapt_path = os.path.join(result_folder, f"{args.pyapt_id}__{now.strftime('%H_%M_%S')}")
    with open(pyapt_path, "a") as t_f:
        t_f.write(" ")

    val_dataset, input_size = get_dataset.get_dataset(
        args.val_dataset,
        split=args.val_split,
        meta={"version": args.version, "split_mode": args.split_mode},
        use_cache=args.use_cache,
        mini_factor=args.mini_factor,
        mode=args.mode,
        fraction=args.fraction,
        no_augm=True,
        center_idx=args.center_idx,
        scale_jittering=0,
        center_jittering=0,
        sample_nb=None,
        has_dist2strong=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False,
        collate_fn=collate.meshreg_collate,
    )

    opts = reloadmodel.load_opts(args.resume)
    model, epoch = reloadmodel.reload_model(args.resume, opts)
    freeze.freeze_batchnorm_stats(model)  # Freeze batchnorm

    all_samples = []

    model.eval()
    model.cuda()
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        all_results = []
        with torch.no_grad():
            loss, results, losses = model(batch)
            all_results.append(results)

        for img_idx, img in enumerate(batch[BaseQueries.IMAGE]):    # Each batch has 4 images
            network_out = all_results[0]
            sample_idx = batch['idx'][img_idx]
            handpose, handtrans, handshape = val_dataset.pose_dataset.get_hand_info(sample_idx)

            sample_dict = dict()
            # sample_dict['image'] = img
            sample_dict['obj_faces'] = batch[BaseQueries.OBJFACES][img_idx, :, :]
            sample_dict['obj_verts_gt'] = batch[BaseQueries.OBJVERTS3D][img_idx, :, :]
            sample_dict['hand_faces'], _ = manoutils.get_closed_faces()
            sample_dict['hand_verts_gt'] = batch[BaseQueries.HANDVERTS3D][img_idx, :, :]

            sample_dict['obj_verts_pred'] = network_out['recov_objverts3d'][img_idx, :, :]
            sample_dict['hand_verts_pred'] = network_out['recov_handverts3d'][img_idx, :, :]
            # sample_dict['hand_adapt_trans'] = network_out['mano_adapt_trans'][img_idx, :]
            sample_dict['hand_pose_pred'] = network_out['pose'][img_idx, :]
            sample_dict['hand_beta_pred'] = network_out['shape'][img_idx, :]
            sample_dict['side'] = batch[BaseQueries.SIDE][img_idx]

            sample_dict['hand_pose_gt'] = handpose
            sample_dict['hand_beta_gt'] = handshape
            sample_dict['hand_trans_gt'] = handtrans
            sample_dict['hand_extr_gt'] = torch.Tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            for k in sample_dict.keys():
                sample_dict[k] = to_cpu_npy(sample_dict[k])

            all_samples.append(sample_dict)

    print('Saving final dict', len(all_samples))
    with open('all_samples.pkl', 'wb') as handle:
        pickle.dump(all_samples, handle)
    print('Done saving')


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    # torch.multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser()
    parser.add_argument("--com", default="debug/")

    parser.add_argument("--split_mode", default="paper", choices=["objects", "paper"],
                        help="HO3D possible splits, 'paper' for hand baseline, 'objects' for photometric consistency")

    # Dataset params
    parser.add_argument("--val_dataset", choices=["ho3dv2"], default="ho3dv2")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--mini_factor", type=float, default=1)
    parser.add_argument("--max_verts", type=int, default=1000)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--synth", action="store_true")
    parser.add_argument("--version", default=3, type=int)
    parser.add_argument("--fraction", type=float, default=1)
    parser.add_argument("--mode", choices=["strong", "weak", "full"], default="strong")

    # Test options
    parser.add_argument("--dump_results", action="store_true")
    parser.add_argument("--render_results", action="store_true")
    parser.add_argument("--render_freq", type=int, default=10)

    # Model params
    parser.add_argument("--center_idx", default=9, type=int)
    parser.add_argument(
        "--true_root", action="store_true", help="Replace predicted wrist position with ground truth root"
    )
    parser.add_argument("--resume")

    # Training params
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--freeze_batchnorm", action="store_true")
    parser.add_argument("--pyapt_id")
    parser.add_argument("--criterion2d", choices=["l2", "l1", "smooth_l1"], default="l2")

    # Weighting
    parser.add_argument("--obj_trans_factor", type=float, default=1)
    parser.add_argument("--obj_scale_factor", type=float, default=1)

    # Evaluation params
    parser.add_argument("--mask_threshold", type=float, default=0.9)
    parser.add_argument("--json_folder", default="jsonres/res")

    # Weighting params
    parser.add_argument("--display_freq", type=int, default=100)
    parser.add_argument("--snapshot", type=int, default=50)

    args = parser.parse_args()
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)
