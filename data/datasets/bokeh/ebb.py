#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import os
# from torchvision.datasets import ImageFolder
from typing import Optional, Tuple, Dict, List, Union
import torch
import argparse

from utils import logger

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T
from ...collate_fns import register_collate_fn


@register_dataset(name="ebb", task="bokeh")
class EBBDataset(BaseImageDataset):
    """
    ImageNet Classification Dataset that uses PIL for reading and augmenting images. The dataset structure should
    follow the ImageFolder class in :class:`torchvision.datasets.imagenet`

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False

    .. note::
        We recommend to use this dataset class over the imagenet_opencv.py file.

    """

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        BaseImageDataset.__init__(
            self, opts=opts, is_training=is_training, is_evaluation=is_evaluation
        )
        root = self.root
        list_dir = os.path.join(root, "list")

        if self.is_training:
            # use the EBB! train data and validation data ??
            data_file = os.path.join(list_dir, "train.txt")
        else:
            # use the EBB! validation data ??
            data_file = os.path.join(list_dir, "valid.txt")

        self.images = []
        self.bokehs = [] # let ground truth named bokeh
        with open(data_file, "r") as lines:
            for line in lines:
                line_split = line.split(" ")
                rgb_img_loc = root + os.sep + line_split[0].strip()
                assert os.path.isfile(
                    rgb_img_loc
                ), "RGB file does not exist at: {}".format(rgb_img_loc)

                bokeh_img_loc = root + os.sep + line_split[1].strip()
                assert os.path.isfile(
                    bokeh_img_loc
                ), "Mask image does not exist at: {}".format(bokeh_img_loc)
                self.images.append(rgb_img_loc)
                self.bokehs.append(bokeh_img_loc)

        self.ignore_label = 255
        self.bgrnd_idx = 0


    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--dataset.ebb.crop-ratio",
            type=float,
            default=0.875,
            help="Crop ratio",
        )
        return parser

    def _training_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
            Training data augmentation methods.
                Image --> RandomResizedCrop --> RandomHorizontalFlip --> Optional(AutoAugment or RandAugment)
                --> Tensor --> Optional(RandomErasing) --> Optional(MixUp) --> Optional(CutMix)

        .. note::
            1. AutoAugment and RandAugment are mutually exclusive.
            2. Mixup and CutMix are applied on batches are implemented in trainer.

            For bokeh, i will remove all the options.
        """
        aug_list = [
            # T.RandomResizedCrop(opts=self.opts, size=size),
            T.Resize(opts=self.opts),
            T.RandomHorizontalFlip(opts=self.opts),
        ]

        aug_list.append(T.ToTensor(opts=self.opts))

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
        Same with training transformation.
        """
        aug_list = [T.Resize(opts=self.opts), T.ToTensor(opts=self.opts)]
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _evaluation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
        Evaluation augmentation
            Image --> Resize --> CenterCrop --> ToTensor
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        crop_size = (crop_size_h, crop_size_w)

        if self.is_evaluation:
            transform_fn = self._evaluation_transforms(size=crop_size)
        else:
            # same for train and validation
            transform_fn = self._training_transforms(size=crop_size)

        input_img = self.read_image_pil(self.images[img_index])
        target_img = self.read_image_pil(self.bokehs[img_index])

        if input_img is None:
            # Sometimes images are corrupt
            # Skip such images
            logger.log("Img index {} is possibly corrupt.".format(img_index))
            input_tensor = torch.zeros(
                size=(3, crop_size_h, crop_size_w), dtype=self.img_dtype
            )
            target = -1
            data = {"image": input_tensor}
            data["bokeh"] = input_tensor
        else:
            data = {"image": input_img}
            data["bokeh"] = target_img
            data = transform_fn(data)

        data["label"] = data["bokeh"]
        del data["bokeh"]

        # data["sample_id"] = img_index

        return data

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.images),
            transforms_str,
        )


@register_collate_fn(name="ebb_collate_fn")
def ebb_collate_fn(batch: List, opts) -> Dict:
    batch_size = len(batch)
    img_size = [batch_size, *batch[0]["image"].shape]
    img_dtype = batch[0]["image"].dtype

    images = torch.zeros(size=img_size, dtype=img_dtype)
    # fill with -1, so that we can ignore corrupted images
    labels = torch.full(size=[batch_size], fill_value=-1, dtype=torch.long)
    sample_ids = torch.zeros(size=[batch_size], dtype=torch.long)
    valid_indexes = []
    for i, batch_i in enumerate(batch):
        label_i = batch_i.pop("label")
        images[i] = batch_i.pop("image")
        labels[i] = label_i  # label is an int
        sample_ids[i] = batch_i.pop("sample_id")  # sample id is an int
        if label_i != -1:
            valid_indexes.append(i)

    valid_indexes = torch.tensor(valid_indexes, dtype=torch.long)
    images = torch.index_select(images, dim=0, index=valid_indexes)
    labels = torch.index_select(labels, dim=0, index=valid_indexes)
    sample_ids = torch.index_select(sample_ids, dim=0, index=valid_indexes)

    channels_last = getattr(opts, "common.channels_last", False)
    if channels_last:
        images = images.to(memory_format=torch.channels_last)

    return {
        "image": images,
        "label": labels,
        "sample_id": sample_ids,
        "on_gpu": images.is_cuda,
    }
