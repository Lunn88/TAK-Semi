import os.path
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from scipy.ndimage import zoom
from collections import defaultdict


def get_loc_mask(volume_batch, cube_size=32):

    # features: 4, 1, 96, 96, 96
    bs, c, w, h, d = volume_batch

    # sx: 3, sy: 3, sz: 3
    sx = math.ceil((w - cube_size) / cube_size) + 1
    sy = math.ceil((h - cube_size) / cube_size) + 1
    sz = math.ceil((d - cube_size) / cube_size) + 1

    loc_list = []
    for x in range(1, sx + 1):
        for y in range(1, sy + 1):
            for z in range(1, sz + 1):
                # 27 patches : 0 ~ 26
                # l(x, y, z) = x + Wy +WHz
                # e.g.: when x = 1, y = 1, z = 1, sx = sy = sz = 3
                # loc_value = 0 + 3 * 0 + 3 * 3 * 0 = 13
                # e.g.: when x = 3, y = 3, z = 3,
                # loc_value = 2 + 3 * 2 + 3 * 3 * 2 = 26
                loc_value = torch.tensor((x-1) + sx * (y-1) + sx * sy * (z-1))
                loc_list.append(loc_value.unsqueeze(0))

    # loc_list: N=27 x [1, 1]
    # tensor([0]), tensor([9]), tensor([18]), tensor([3]), tensor([12]), tensor([21]), tensor([6]), tensor([15]), tensor([24]), tensor([1]), tensor([10]), tensor([19]), tensor([4]), tensor([13]), tensor([22]), tensor([7]), tensor([16]), tensor([25]), tensor([2]), tensor([11]), tensor([20]), tensor([5]), tensor([14]), tensor([23]), tensor([8]), tensor([17]), tensor([26])
    return loc_list


def get_part_and_rec_ind(volume_shape, nb_cubes, nb_chnls, rand_loc_ind=None):
    bs, c, w, h, d = volume_shape # 4, 1, 96, 96, 96
    

    # partition
    if rand_loc_ind is None:
        rand_loc_ind = torch.argsort(torch.rand(bs, nb_cubes, nb_cubes, nb_cubes), dim=0).cuda()
        # print(rand_loc_ind.shape, torch.unique(rand_loc_ind, return_counts=True))
        # torch.Size([4, 3, 3, 3]) (tensor([0, 1, 2, 3], device='cuda:0'), tensor([27, 27, 27, 27], device='cuda:0'))
        cube_part_ind = rand_loc_ind.view(bs, 1, nb_cubes, nb_cubes, nb_cubes) # torch.Size([4, 1, 3, 3, 3])
        cube_part_ind = cube_part_ind.repeat_interleave(c, dim=1) #  1 torch.Size([4, 1, 3, 3, 3])
        cube_part_ind = cube_part_ind.repeat_interleave(w // nb_cubes, dim=2) # 32 torch.Size([4, 1, 96, 3, 3])
        cube_part_ind = cube_part_ind.repeat_interleave(h // nb_cubes, dim=3) # 32 torch.Size([4, 1, 96, 96, 3])
        cube_part_ind = cube_part_ind.repeat_interleave(d // nb_cubes, dim=4) # 32 torch.Size([4, 1, 96, 96, 96])
    else:
        cube_part_ind = None
    # recovery
    rec_ind = torch.argsort(rand_loc_ind, dim=0).cuda()
    #print(rec_ind)
    cube_rec_ind = rec_ind.view(bs, 1, nb_cubes, nb_cubes, nb_cubes)
    cube_rec_ind = cube_rec_ind.repeat_interleave(nb_chnls, dim=1)   # 16 torch.Size([4, 16, 3, 3, 3])
    cube_rec_ind = cube_rec_ind.repeat_interleave(w // nb_cubes, dim=2) # 32 torch.Size([4, 16, 96, 3, 3])
    cube_rec_ind = cube_rec_ind.repeat_interleave(h // nb_cubes, dim=3)
    cube_rec_ind = cube_rec_ind.repeat_interleave(d // nb_cubes, dim=4)

    return cube_part_ind, cube_rec_ind, rand_loc_ind
    # print(cube_part_ind.shape, cube_rec_ind.shape) 
    # print(torch.unique(cube_part_ind, return_counts=True))
    # print(torch.unique(cube_rec_ind, return_counts=True))
    # [4, 1, 96, 96, 96] [4, 16, 96, 96, 96]
    # (tensor([0, 1, 2, 3], device='cuda:0'), tensor([884736, 884736, 884736, 884736], device='cuda:0'))
    # (tensor([0, 1, 2, 3], device='cuda:0'), tensor([14155776, 14155776, 14155776, 14155776], device='cuda:0'))


def get_random_ori_mask(volume_shape):
    bs, c, w, h, d = volume_shape
    rand_ori_ind = torch.argsort(torch.rand(bs, 1, 1, 1), dim=0)
    rand_ori_ind = rand_ori_ind.view(bs, 1, 1, 1, 1)
    rand_ori_ind = rand_ori_ind.repeat_interleave(c, dim=1)
    rand_ori_ind = rand_ori_ind.repeat_interleave(w, dim=2)
    rand_ori_ind = rand_ori_ind.repeat_interleave(h, dim=3)
    rand_ori_ind = rand_ori_ind.repeat_interleave(d, dim=4)
    return rand_ori_ind


def get_one_cube_swap_ind(volume_shape, patch_pixel_shape):
    bs, c, w, h, d = volume_shape
    rand_ori_ind = get_random_ori_mask(volume_shape)

    # Get sub-volume's start-point
    patch_pixel_x, patch_pixel_y, patch_pixel_z = patch_pixel_shape
    assert patch_pixel_x < w and patch_pixel_y < h and patch_pixel_z < d, "out of length"
    patch_pixel_shape_new = (bs, c, patch_pixel_x, patch_pixel_y, patch_pixel_z)
    rand_sub_ind = get_random_ori_mask(patch_pixel_shape_new)

    w_start = torch.randint(low=0, high=w - patch_pixel_x, size=(1, 1))
    h_start = torch.randint(low=0, high=h - patch_pixel_y, size=(1, 1))
    d_start = torch.randint(low=0, high=d - patch_pixel_z, size=(1, 1))

    rand_ori_ind[:, :, w_start:w_start+patch_pixel_x, h_start:h_start+patch_pixel_y, d_start:d_start+patch_pixel_z] = rand_sub_ind
    rand_ori_ind = rand_ori_ind.cuda()

    return rand_ori_ind


def get_one_cube_rec(ori_ind, nb_chnls):
    bs, c, w, h, d = ori_ind.shape
    rec_ind = torch.argsort(ori_ind, dim=0).cuda()
    cube_rec_ind = rec_ind.view(bs, c, w, h, d)
    cube_rec_ind = cube_rec_ind.repeat_interleave(nb_chnls, dim=1)
    return cube_rec_ind


class OrganClassLogger:
    def __init__(self, num_classes=14):
        self.num_classes = num_classes
        self.class_total_pixel_store = []

        # e.g.: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.class_dist_init = [0 for i in range(self.num_classes)]

    def append_class_list(self, dist_value):
        self.class_total_pixel_store.append(dist_value)
        # print('append', len(self.class_total_pixel_store), dist_value.shape)

    def update_class_dist(self):
        #print(self.class_total_pixel_store)
        class_total_list = torch.cat(self.class_total_pixel_store, 0)
        # print('update ', class_total_list.shape)
        for class_ind in range(self.num_classes):
            class_row_inds = torch.where(class_total_list == class_ind)[0]
            # print(class_ind, len(class_row_inds))
            self.class_dist_init[class_ind] = len(class_row_inds)
        self.class_total_pixel_store = []

    def get_class_dist(self, normalize=False):
        if isinstance(self.class_dist_init, list):
            class_dist = torch.Tensor(self.class_dist_init).float()
        else:
            class_dist = self.class_dist_init.float()

        if normalize:
            class_dist = class_dist / class_dist.sum()
        return class_dist

