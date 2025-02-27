import torch
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import h5py


class AMOS(Dataset):
    """ AMOS Dataset """
    def __init__(self, image_list, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = image_list

        self.images = []
        self.laebls = []
        for i in range(len(self.image_list)):
            image_name = self.image_list[i] 
            image_path = self._base_dir + '{}_image.npy'.format(image_name)
            label_path = self._base_dir + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
    
            self.images.append(image)
            self.laebls.append(label)
            print('Loading ', image_path)
            
        print("Loaded! Total {} samples for training".format(len(self.image_list)))



    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):        
        image = self.images[idx]
        label = self.laebls[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)        
        return sample

class AMOS_fast(Dataset):
    """ AMOS Dataset """
    def __init__(self, labeled_list, unlabeled_list, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.labeled_list = labeled_list
        self.unlabeled_list = unlabeled_list
        
        self.images_l = []
        self.laebls_l = []
        for i in range(len(self.labeled_list)):
            image_name = self.labeled_list[i] 
            image_path = self._base_dir + '{}_image.npy'.format(image_name)
            label_path = self._base_dir + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
            self.images_l.append(image)
            self.laebls_l.append(label)
            print('Loading {:2d}-th labeled sample from {}'.format(i, image_path))

        self.images_u = []
        self.laebls_u = []
        for i in range(len(self.unlabeled_list)):
            image_name = self.unlabeled_list[i] 
            image_path = self._base_dir + '{}_image.npy'.format(image_name)
            label_path = self._base_dir + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
            self.images_u.append(image)
            self.laebls_u.append(label)
            print('Loading {:2d}-th unlabeled sample from {}'.format(i, image_path))
            
        print("Loaded! Total {} samples for training".format(len(labeled_list + unlabeled_list)))
        print(len(self.images_l), len(self.laebls_l), len(self.images_u), len(self.laebls_u))


    def __len__(self):
        return (len(self.unlabeled_list)*4)

    def __getitem__(self, idx):
        if idx < len(self.unlabeled_list)*2: # labeled data
            idx = idx % len(self.labeled_list)
            image = self.images_l[idx]
            label = self.laebls_l[idx]
        else:                              # unlabeled data
            idx = idx % len(self.unlabeled_list)
            image = self.images_u[idx]
            label = self.laebls_u[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, is_2d=False):
        self.is_2d = is_2d

    def __call__(self, sample):
        # image, label: For AHNet 2D to 3D,
        # 3D: WxHxD -> 1xWxHxD, 96x96x96 -> 1x96x96x96
        # 2D: WxHxD -> CxWxh, 224x224x3 -> 3x224x224
        image, label = sample['image'], sample['label']

        if self.is_2d:
            image = image.transpose(2, 0, 1).astype(np.float32)
            label = label.transpose(2, 0, 1)[1, :, :]
        else:
            # image = image.transpose(1, 0, 2)
            # label = label.transpose(1, 0, 2)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
