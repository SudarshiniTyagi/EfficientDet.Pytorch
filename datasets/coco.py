from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image
import torch
import torchvision


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations',
                                      'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images',
                            self.set_name, image_info['file_name'])
        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


def read_image_mat(file_name):
    data = h5py.File(file_name, "r")
    image = np.array(data["image"]).T
    data.close()
    return image

class CustomCocoDetection(torchvision.datasets.coco.CocoDetection):
    """
    Custom version of Coco dataset that loads from hdf5 instead
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]

        image_level = coco.loadImgs(img_id)[0].get("image_level", 0)

        img = Image.fromarray(
            read_image_mat(os.path.join(self.root, path)) / 16
        ).convert("RGB")
        best_center = coco.loadImgs(img_id)[0].get("best_center", (0, 0))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class BreastDataset(CustomCocoDetection):
    def __init__(self, ann_file=None, root=None, set_type="train", transforms=None):
        if set_type == "train":
            root = "/gpfs/data/geraslab/jp4989/data/2010_2017_cropped_images_hdf5/"
            ann_file = "breasts/train_with_unknowns.json"
        elif set_type == "test":
            root = "/gpfs/data/geraslab/jp4989/data/2010_2017_cropped_images_hdf5/",
            ann_file = "/gpfs/data/geraslab/sudarshini/rcnn_with_attention/datasets/breasts/test_with_unknowns.json"
        elif set_type == "val":
            root = "/gpfs/data/geraslab/jp4989/data/2010_2017_cropped_images_hdf5/",
            ann_file = "/gpfs/data/geraslab/sudarshini/rcnn_with_attention/datasets/breasts/val_with_unknowns.json"
        super(BreastDataset, self).__init__(root, ann_file)
        self.biopsy_image_ratio = float(os.environ.get("BIOPSY_RATIO", 0.5))
        self.max_annotation_size = int(os.environ.get("MAX_ANNOTATION_SIZE", 0))
        self.max_biopsy = 13444
        self.split = ann_file.split("/")[-1].split(".")[
            0
        ]  # Ugly hack to workaround expected format
        self.split = self.split.replace("_with_unknowns", "")

        self.ids = sorted(self.ids)

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.transforms = transforms

    def __getitem__(self, idx):
        if (
            self.split == "train"
            and idx > self.max_biopsy
            and random.random() < self.biopsy_image_ratio
        ):
            idx = random.randrange(self.max_biopsy)

        img, anno = super(BreastDataset, self).__getitem__(
            idx
        )

        if self.max_annotation_size and self.split == "train":
            anno = [a for a in anno if a["bbox"][2] * a["bbox"][3] <= self.max_annotation_size]


        if self.transforms is not None:
            img, target = self.transforms(img, anno)

        return img, anno

    def get_img_info(self, idx):
        img_id = self.id_to_img_map[idx]
        img_data = self.coco.imgs[img_id]
        return img_data
    def num_classes(self):
        return 2

if __name__ == '__main__':
    from augmentation import get_augumentation
    dataset = CocoDataset(root_dir='/root/data/coco', set_name='trainval35k',
                          transform=get_augumentation(phase='train'))
    sample = dataset[0]
    print('sample: ', sample)
