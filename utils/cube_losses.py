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


def unmix_tensor(patch_list, unmix_shape):
    # patch_list: 4, 27, 16, 32, 32, 32
    # unmix_shape: 4, 1, 96, 96, 96

    _, _, w, h, d = unmix_shape
    bs, _, c, cube_sx, cube_sy, cube_sz = patch_list.shape

    # sx: 3, sy: 3, sz: 3
    sx = math.ceil((w - cube_sx) / cube_sx) + 1
    sy = math.ceil((h - cube_sy) / cube_sy) + 1
    sz = math.ceil((d - cube_sz) / cube_sz) + 1
    # res: 4, 16, 96, 96, 96
    res = torch.zeros(bs, c, w, h, d).cuda()

    for x in range(1, sx + 1):
        xs = min(cube_sx * (x - 1), w - cube_sx)
        for y in range(1, sy + 1):
            ys = min(cube_sy * (y - 1), h - cube_sy)
            for z in range(1, sz + 1):
                zs = min(cube_sz * (z - 1), d - cube_sz)
                # 27 patches : 0 ~ 26
                # loc_tmp: 0 ~ 26
                loc_tmp = (x - 1) + sx * (y - 1) + sx * sy * (z - 1)
                res[:, :, xs:xs + cube_sx, ys:ys + cube_sy, zs:zs + cube_sz] = patch_list[:, loc_tmp, :]
    # res: 4, 16, 96, 96, 96
    return res


def get_patch_list(volume_batch, cube_size=32):

    # features: 4, 1, 96, 96, 96
    bs, c, w, h, d = volume_batch.shape
    h_ = h // cube_size * cube_size
    w_ = w // cube_size * cube_size
    d_ = d // cube_size * cube_size

    # sx: 3, sy: 3, sz: 3
    sx = math.ceil((w_ - cube_size) / cube_size) + 1
    sy = math.ceil((h_ - cube_size) / cube_size) + 1
    sz = math.ceil((d_ - cube_size) / cube_size) + 1

    patch_list = []
    for x in range(1, sx + 1):
        xs = min(cube_size * (x - 1), w_ - cube_size)
        for y in range(1, sy + 1):
            ys = min(cube_size * (y - 1), h_ - cube_size)
            for z in range(1, sz + 1):
                zs = min(cube_size * (z - 1), d_ - cube_size)
                # 27 patches : 0 ~ 26
                # add patch_number dimension: 4, 1, 32, 32, 32 -> 4, 1, 1, 32, 32, 32
                img_patch = volume_batch[:, :, xs:xs + cube_size, ys:ys + cube_size, zs:zs + cube_size]
                patch_list.append(img_patch.unsqueeze(1))

    # patch_list: N=27 x [4, 1, 1, 32, 32, 32] (bs, pn, c, w, h, d)
    return patch_list


def cube_location_loss(model, loc_list, patch_list, idx, do_contrast):

    # patch_list: N=27 x [4, 1, 1, 32, 32, 32] -> 4x27x1x32x32x32
    patches = torch.cat(patch_list, dim=1) # torch.Size([4, 27, 1, 32, 32, 32])
    bs = patches.shape[0]

    # 27
    loc_mask = torch.cat(loc_list, dim=0).cuda()

    loc_loss = 0
    feat_list = []
    feat_embed_list = []

    for i in range(bs):
        # patches -> 27x1x32x32x32
        # feat_patch: [f1, f2, f3, f4, f5], f5 = 27x256x2x2x2
        feat_patch, feat_embed_patch = model.forward_encoder(patches[i, :], do_contrast)[:2]
        # print(i, len(feat_patch), feat_patch[-1].shape)

        feat_list.append(feat_patch)
        feat_embed_list.append(feat_embed_patch)

        # 27x256x2x2x2 -> 27x2048
        feat_flatten = torch.flatten(feat_patch[-1], start_dim=1, end_dim=4)
        feat_tmp = feat_flatten[idx, :].view(feat_flatten.size())
        loc_mask_tmp = loc_mask[idx].view(loc_mask.size())
        loc_pred = model.fc_layer(feat_tmp)
        # loc_pred: 27, 27 loc_mask_tmp: 27
        loc_pred = F.log_softmax(loc_pred, dim=1)
        loc_loss += F.nll_loss(loc_pred, loc_mask_tmp)
    # feat_list: 4 x [f1, f2, f3, f4, f5]
    
    return loc_loss / bs, feat_list, feat_embed_list


def get_mix_pl(ema_model, unlabeled_feat_list, text_embedding, ori_shape, unlabeled_bs=2):

    # unlabeled_feat_list: 2x[f1, f2, f3, f4, f5], f5=27x256x2x2x2
    with torch.no_grad():
        pred_list = []

        for i in range(unlabeled_bs):
            # pred_tmp: [f1-f5] -> 27x9x32x32x32
            # pred_tmp = ema_model.forward_decoder(feat_list[i])[0].detach_()

            embedding_tmp = ema_model.forward_decoder(unlabeled_feat_list[i]).detach()
            pred_tmp = ema_model.forward_prediction_head(unlabeled_feat_list[i], text_embedding, embedding_tmp)

            pred_list.append(pred_tmp)
        pred_all = torch.stack(pred_list, dim=0) # 2, 27, 16, 32, 32, 32
        un_pl = unmix_tensor(pred_all, ori_shape) # 2, 16, 96, 96, 96

    return un_pl


# def get_mix_embed(ema_model, feat_list, ori_shape, unlabeled_bs=2):
#
#     # feat_list: 2x[f1, f2, f3, f4, f5], f5=27x256x2x2x2
#     with torch.no_grad():
#         embed_list = []
#         for i in range(unlabeled_bs):
#             # pred_tmp: [f1-f5] -> 27x9x32x32x32
#             pred_tmp = ema_model.forward_decoder(feat_list[i])[1]
#             embed_list.append(pred_tmp.unsqueeze(0))
#         # pred_all: 2 x [1, 27, 9, 32, 32, 32] -> 2, 27, 9, 32, 32, 32 -> 2, 9, 96, 96, 96
#         embed_all = torch.cat(embed_list, dim=0)
#         un_embed = unmix_tensor(embed_all, ori_shape)
#
#     return un_embed
