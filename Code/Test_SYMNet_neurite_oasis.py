import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import SYMNet,SpatialTransform, DiffeomorphicTransform, CompositionTransform, SpatialTransformNearest
from Functions import generate_grid,save_img,save_flow, load_4D_with_header, imgnorm, Predict_dataset_crop
import glob
import torch.utils.data as Data
import random


parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/SYMNet_neurite_oasis_smo30_update_80000.pth',
                    help="frequency of saving models")
# parser.add_argument("--savepath", type=str,
#                     dest="savepath", default='../Result',
#                     help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
# parser.add_argument("--fixed", type=str,
#                     dest="fixed", default='../Data/image_A_full_size.nii.gz',
#                     help="fixed image")
# parser.add_argument("--moving", type=str,
#                     dest="moving", default='../Data/image_B_full_size.nii.gz',
#                     help="moving image")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../Data/OASIS/neurite-oasis.v1.0',
                    help="data path for training images")
parser.add_argument("-g", type=str,
                    dest="g",
                    default=0)


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice/num_count


def test(device):
    model = SYMNet(2, 3, opt.start_channel).to(device)
    transform = SpatialTransform().to(device)

    diff_transform = DiffeomorphicTransform(time_step=7).to(device)
    com_transform = CompositionTransform().to(device)
    transform_nearest = SpatialTransformNearest().to(device)

    model.load_state_dict(torch.load(opt.modelpath, map_location='cpu'))
    model.eval()
    transform.eval()
    diff_transform.eval()
    com_transform.eval()

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    datapath = opt.datapath
    test_id_list = list(range(260, 410))
    random.shuffle(test_id_list)
    fixed_list = test_id_list[:5]
    moving_list = test_id_list[5:]

    total_img_list = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
    total_segs_list = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))

    dice_total = []
    violation_total = []
    for f in fixed_list:
        fixed_img = total_img_list[f]
        fixed_label = total_segs_list[f]
        imgs = [total_img_list[i] for i in moving_list]
        labels = [total_segs_list[i] for i in moving_list]

        valid_generator = Data.DataLoader(Predict_dataset_crop(fixed_img, imgs, fixed_label, labels, norm=True),
                                            batch_size=1,
                                            shuffle=False, num_workers=2)
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_generator):
                X, Y, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), data['move_label'].to(
                    device), data['fixed_label'].to(device)
                
                F_xy, F_yx = model(X, Y)

                F_X_Y_half = diff_transform(F_xy, grid, range_flow)
                F_Y_X_half = diff_transform(F_yx, grid, range_flow)

                F_X_Y_half_inv = diff_transform(-F_xy, grid, range_flow)
                F_Y_X_half_inv = diff_transform(-F_yx, grid, range_flow)

                F_X_Y = com_transform(F_X_Y_half, F_Y_X_half_inv, grid, range_flow)
                F_Y_X = com_transform(F_Y_X_half, F_X_Y_half_inv, grid, range_flow)

                # F_BA = F_Y_X.permute(0, 2, 3, 4, 1).data.cpu().numpy()[0, :, :, :, :]
                # F_BA = F_BA.astype(np.float32) * range_flow
                
                # F_AB = F_X_Y.permute(0, 2, 3, 4, 1).data.cpu().numpy()[0, :, :, :, :]
                # F_AB = F_AB.astype(np.float32) * range_flow

                X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0, 0, :, :, :]
                Y_label = Y_label.data.cpu().numpy()[0, 0, :, :, :]

                dice_score = dice(np.floor(X_Y_label), np.floor(Y_label))
                dice_total.append(dice_score)

                inverse_compose = com_transform(F_X_Y, F_Y_X, grid, range_flow)
                violation = torch.mean((inverse_compose*range_flow)**2).item()
                violation_total.append(violation)
                
                # warped_B = transform(moved_img, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0, 0, :, :, :]
                # warped_A = transform(fixed_img, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0, 0, :, :, :]

                # save_flow(F_BA, savepath + '/wrapped_flow_B_to_A_full_size.nii.gz')
                # save_flow(F_AB, savepath + '/wrapped_flow_A_to_B_full_size.nii.gz')

                # save_img(warped_B, savepath + '/wrapped_norm_B_to_A_full_size.nii.gz', header=moved_header, affine=moved_affine)
                # save_img(warped_A, savepath + '/wrapped_norm_A_to_B_full_size.nii.gz', header=fixed_header, affine=fixed_affine)
                
                # print("Finished.")
    
    print(f"Dice mean:{np.array(dice_total).mean()}")
    print(f"Dice mean:{np.array(violation_total).mean()}")


if __name__ == '__main__':
    # imgshape = (160, 192, 224)
    imgshape = (160, 144, 192)
    range_flow = 100

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.g}")
    torch.cuda.set_device(device)

    test(device)