from __future__ import print_function, division
import os

import sys
sys.path.append("/home/liujiahui/PycharmProjects/pspnet-pytorch/dataloaders")

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
#from mypath import Path


path = ('/home/liujiahui/data_zoo/ISPRS/Vaihingen')

class ISPRSSegmentation(Dataset):
    """
    ISPRS dataset
    """

    def __init__(self,
                 base_dir=path,
                 split='train',
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir  # ISPRS image dir
        self._image_dir = os.path.join(self._base_dir, 'top')
        self._cat_dir = os.path.join(self._base_dir, 'gt')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'pathlist')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()
                # print(lines)

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".tif")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)

                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return 'ISPRS(split=' + str(self.split) + ')'


if __name__ == '__main__':
    import custom_transforms as tr
    from utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([tr.RandomHorizontalFlip(),
                                                 tr.RandomCrop(512),
                                                 tr.RandomRotate(15),
                                                 tr.ToTensor()]
                                                )

    isprs_train = ISPRSSegmentation(split='train', transform=composed_transforms_tr)

    dataloader = DataLoader(isprs_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            tmp = np.squeeze(tmp, axis=0)
            segmap = decode_segmap(tmp, dataset='ISPRS')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break
    plt.show(block=True)


