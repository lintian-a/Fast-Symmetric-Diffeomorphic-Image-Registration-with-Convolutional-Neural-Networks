import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import SYMNet,SpatialTransform, DiffeomorphicTransform, CompositionTransform, SpatialTransformNearest, JacboianDet
from Functions import generate_grid,save_img,save_flow, load_4D_with_header, imgnorm, Predict_dataset_crop, Predict_dataset
import glob
import torch.utils.data as Data
import random
from oasis_data import get_data_list, extract_id
import torch.nn.functional as F


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


def crop_center(img, cropx, cropy, cropz):
    _,_, x, y, z = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2
    return img[:,:,startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]

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

    fixed_imgs, fixed_segs, moving_imgs, moving_segs = get_data_list()

    dice_total = []
    violation_total = []
    flips_total = []
    with torch.no_grad():
        for f, f_seg in zip(fixed_imgs, fixed_segs):

            valid_generator = Data.DataLoader(Predict_dataset(f, moving_imgs, f_seg, moving_segs, norm=True),
                                                batch_size=1,
                                                shuffle=False, num_workers=2)
            with torch.no_grad():
                for batch_idx, data in enumerate(valid_generator):
                    X, Y, X_label, Y_label = (
                        crop_center(data['move'], *imgshape).to(device), 
                        crop_center(data['fixed'], *imgshape).to(device), 
                        crop_center(data['move_label'], *imgshape).to(device), 
                        crop_center(data['fixed_label'], *imgshape).to(device))
                    
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
                    

                    X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid).cpu().numpy()[0,0]
                    Y_label = Y_label.cpu().numpy()[0, 0, :, :, :]

                    dice_score = dice(X_Y_label, Y_label)

                    dice_total.append(dice_score)

                    inverse_compose = com_transform(F_X_Y, F_Y_X, grid, range_flow)
                    violation = torch.mean(torch.sqrt(torch.sum((inverse_compose*range_flow)**2, dim=1))).item()
                    violation_total.append(violation)

                    flips_total.append((JacboianDet(F_X_Y.permute(0,2,3,4,1)*range_flow, grid)<0).float().mean().item()*100.)
                    
                    # warped_B = transform(moved_img, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0, 0, :, :, :]
                    # warped_A = transform(fixed_img, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0, 0, :, :, :]

                    # save_flow(F_BA, savepath + '/wrapped_flow_B_to_A_full_size.nii.gz')
                    # save_flow(F_AB, savepath + '/wrapped_flow_A_to_B_full_size.nii.gz')

                    # save_img(warped_B, savepath + '/wrapped_norm_B_to_A_full_size.nii.gz', header=moved_header, affine=moved_affine)
                    # save_img(warped_A, savepath + '/wrapped_norm_A_to_B_full_size.nii.gz', header=fixed_header, affine=fixed_affine)
                    
                    # print("Finished.")
    
    print(f"Dice mean:{np.array(dice_total).mean()}")
    print(f"Violation to id mean:{np.array(violation_total).mean()}")
    print(f"Flips(%) mean:{np.array(flips_total).mean()}")


if __name__ == '__main__':
    # imgshape = (160, 192, 224)
    imgshape = (160, 144, 192)
    range_flow = 100

    interpolate_first = False

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.g}")
    torch.cuda.set_device(device)

    test(device)