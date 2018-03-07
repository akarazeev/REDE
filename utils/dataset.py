# File name: dataset.py
# Original file: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
# Edited by: Anton Karazeev <anton.karazeev@gmail.com>
#
# This file is part of REDE project (https://github.com/akarazeev/REDE)

from __future__ import print_function
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import pickle


class REDE(data.Dataset):
    """`REDE` Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/full.pt``
            exists.
        train (bool, optional): If True, creates dataset for training,
            otherwise from testing.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        test_size (float, optional): A portion of the whole dataset that will be used
            for testing.
        test_indices (list, optional): List of indices that correspond to samples
            from test dataset.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    urls = [
        'https://github.com/akarazeev/REDE/raw/master/data/rede/raw/1056-5-parameters.pkl',
        'https://github.com/akarazeev/REDE/raw/master/data/rede/raw/1056-62-111-images.pkl',
        'https://github.com/akarazeev/REDE/raw/master/data/rede/raw/1056-x-frequencies_modes.pkl'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    full_file = 'full.pt'

    def __init__(self, root, train=True, transform=None, test_size=0.2, test_indices=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.test_size = test_size

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.full_images, self.full_parameters = torch.load(
            os.path.join(self.root, self.processed_folder, self.full_file))

        self.train_indices, self.test_indices = train_test_split(np.arange(len(self.full_images)), test_size=self.test_size)

        if not train:
            # Test dataset.
            error_message = 'Pass indices for test from train_dataset with correct length ({})'.format(len(self.test_indices))
            if test_indices is not None:
                # `test_indices` are passed.
                if len(self.test_indices) == len(test_indices):
                    self.test_indices = test_indices
                else:
                    raise RuntimeError(error_message)
            else:
                raise RuntimeError(error_message)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, parameters) where parameter is a tuple of (gap,
                width1, height, radius1, width2). E.g. (2.500000e-07, 0.000001,
                7.000000e-07, 0.000018, 8.000000e-07)
        """
        if self.train:
            img, parameters = self.full_images[self.train_indices[index]], self.full_parameters[self.train_indices[index]]
        else:
            img, parameters = self.full_images[self.test_indices[index]], self.full_parameters[self.test_indices[index]]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image.
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, parameters.type_as(torch.FloatTensor())

    def __len__(self):
        # return len(self.full_images)
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.full_file))

    def download(self):
        """Download the REDE data if it doesn't exist in `processed_folder` already."""
        from six.moves import urllib

        if self._check_exists():
            return

        # Make directories.
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # Download dataset.
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        # Process and save as torch files.
        print('Processing...')

        full_set = (
            read_file(os.path.join(self.root, self.raw_folder, '1056-62-111-images.pkl')),
            read_file(os.path.join(self.root, self.raw_folder, '1056-5-parameters.pkl'))
            # read_file(os.path.join(self.root, self.raw_folder, '1056-x-frequencies_modes.pkl'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.full_file), 'wb') as f:
            torch.save(full_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {} (test_size: {})\n'.format(tmp, self.test_size)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def read_file(path):
    with open(path, 'rb') as f:
        parsed = pickle.load(f)
        return torch.from_numpy(parsed)
