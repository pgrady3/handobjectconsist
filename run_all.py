import argparse
import os

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from tqdm import tqdm

from libyana.exputils.argutils import save_args
from libyana.modelutils import freeze
from libyana.randomutils import setseeds

from meshreg.datasets import collate
from meshreg.datasets.queries import BaseQueries
from meshreg.models.meshregnet import MeshRegNet
from meshreg.netscripts import reloadmodel, get_dataset
from meshreg.neurender import fastrender
from meshreg.visualize import vizdemo
from meshreg.visualize import samplevis
from meshreg.models import manoutils
import pickle


def to_cpu_npy(elem):
    if isinstance(elem, torch.Tensor):
        elem = elem.detach().cpu().numpy()

    return elem


def main(args):
    setseeds.set_all_seeds(args.manual_seed)
    # Initialize hosting
    exp_id = f"checkpoints/{args.dataset}/" f"{args.com}"

    # Initialize local checkpoint folder
    print(f"Saving info about experiment at {exp_id}")
    save_args(args, exp_id, "opt")
    render_folder = os.path.join(exp_id, "images")
    os.makedirs(render_folder, exist_ok=True)
    # Load models
    models = []
    for resume in args.resumes:
        print('Resuming', resume)
        opts = reloadmodel.load_opts(resume)
        model, epoch = reloadmodel.reload_model(resume, opts)
        models.append(model)
        freeze.freeze_batchnorm_stats(model)  # Freeze batchnorm

    dataset, input_res = get_dataset.get_dataset(args.dataset, split=args.split, meta={"version": args.version, "split_mode": args.split_mode},
                                                 mode=args.mode, use_cache=args.use_cache,
                                                 no_augm=True, center_idx=opts["center_idx"], sample_nb=None)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers), drop_last=False,
                                         collate_fn=collate.meshreg_collate)


    all_samples = []

    # Put models on GPU and evaluation mode
    for model in models:
        model.cuda()
        model.eval()

    i = 0
    for batch in tqdm(loader):  # Loop over batches
        all_results = []
        # Compute model outputs
        with torch.no_grad():
            for model in models:
                _, results, _ = model(batch)
                all_results.append(results)

        # Densely render error map for the meshes
        # for results in all_results:
        #     render_results, cmap_obj = fastrender.comp_render(
        #         batch, all_results, rotate=True, modes=("all", "obj", "hand"), max_val=args.max_val
        #     )
        
        # if i > 100:
        #     break
        # i += 1

        for img_idx, img in enumerate(batch[BaseQueries.IMAGE]):    # Each batch has 4 images
            network_out = all_results[0]
            sample_idx = batch['idx'][img_idx]
            print(dataset.pose_dataset.get_hand_info(sample_idx))

            sample_dict = dict()
            # sample_dict['image'] = img
            sample_dict['obj_faces'] = batch[BaseQueries.OBJFACES][img_idx, :, :]
            sample_dict['obj_verts_gt'] = batch[BaseQueries.OBJVERTS3D][img_idx, :, :]
            sample_dict['hand_faces'], _ = manoutils.get_closed_faces()
            sample_dict['hand_verts_gt'] = batch[BaseQueries.HANDVERTS3D][img_idx, :, :]

            sample_dict['obj_verts_pred'] = network_out['recov_objverts3d'][img_idx, :, :]
            sample_dict['hand_verts_pred'] = network_out['recov_handverts3d'][img_idx, :, :]
            # sample_dict['hand_adapt_trans'] = network_out['mano_adapt_trans'][img_idx, :]
            sample_dict['hand_pose'] = network_out['pose'][img_idx, :]
            sample_dict['hand_beta'] = network_out['shape'][img_idx, :]
            sample_dict['side'] = batch[BaseQueries.SIDE][img_idx]

            for k in sample_dict.keys():
                sample_dict[k] = to_cpu_npy(sample_dict[k])

            all_samples.append(sample_dict)

            continue


            # obj_verts_gt = samplevis.get_check_none(sample, BaseQueries.OBJVERTS3D, cpu=False)
            # hand_verts_gt = samplevis.get_check_none(sample, BaseQueries.HANDVERTS3D, cpu=False)
            # hand_verts = samplevis.get_check_none(results, "recov_handverts3d", cpu=False)
            # obj_verts = samplevis.get_check_none(results, "recov_objverts3d", cpu=False)
            # obj_faces = samplevis.get_check_none(sample, BaseQueries.OBJFACES, cpu=False).long()
            # hand_faces, _ = manoutils.get_closed_faces()


            # plt.imshow(img)
            # plt.show()

            for k in batch.keys():
                elem = batch[k]
                # if isinstance(elem, list):
                #     s = len(s)
                if isinstance(elem, torch.Tensor):
                    s = elem.shape
                else:
                    s = elem
                print('{}: Shape {}'.format(k, s))

            for k in all_results[0].keys():
                elem = all_results[0][k]
                if isinstance(elem, list):
                    s = len(s)
                elif isinstance(elem, torch.Tensor):
                    s = elem.shape
                else:
                    s = elem
                print('Network out {}: Shape {}'.format(k, s))

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_trisurf(sample_dict['obj_verts_gt'][:,0], sample_dict['obj_verts_gt'][:,1], sample_dict['obj_verts_gt'][:,2],
                            triangles=sample_dict['obj_faces'])

            ax.plot_trisurf(sample_dict['hand_verts_gt'][:,0], sample_dict['hand_verts_gt'][:,1], sample_dict['hand_verts_gt'][:,2],
                            triangles=sample_dict['hand_faces'])

            ax.plot_trisurf(sample_dict['obj_verts_pred'][:,0], sample_dict['obj_verts_pred'][:,1], sample_dict['obj_verts_pred'][:,2],
                            triangles=sample_dict['obj_faces'])

            ax.plot_trisurf(sample_dict['hand_verts_pred'][:,0], sample_dict['hand_verts_pred'][:,1], sample_dict['hand_verts_pred'][:,2],
                            triangles=sample_dict['hand_faces'])

            plt.show()

    print('Saving final dict', len(all_samples))
    with open('all_samples.pkl', 'wb') as handle:
        pickle.dump(all_samples, handle)
    print('Done saving')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--com", default="debug/", help="Prefix for experimental results")
    parser.add_argument("--manual_seed", default=1, help="Fixed random seed")

    # Dataset params
    parser.add_argument("--dataset", default="ho3dv2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--version", default=2, type=int, help="Version of HO3D dataset to use")
    parser.add_argument("--split_mode", default="objects", choices=["objects", "paper"], help="HO3D possible splits, 'paper' for hand baseline, 'objects' for photometric consistency")
    parser.add_argument("--mode", default="full", help="[viz|full], 'viz' for selected dataset samples, 'full' for random ones")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--max_val", default=0.1, type=float, help="Max value (in meters) for colormap error range")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--workers", default=0, type=int)

    # Model parameters
    parser.add_argument("--resumes", nargs="+", default=["releasemodels/fphab/hands_and_objects/checkpoint_200.pth"])
    parser.add_argument("--model_names", nargs="+", default=["Supervised data: 100%"])

    # Loss parameters
    parser.add_argument("--criterion2d", choices=["l2", "l1", "smooth_l1"], default="l2")
    parser.add_argument("--display_freq", type=int, default=500, help="How often to generate visualizations (training steps)")

    args = parser.parse_args()
    args.model_names = ["Ground Truth"] + args.model_names
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)
