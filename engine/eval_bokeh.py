#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os.path
import numpy as np
import torch
import multiprocessing
from torch.cuda.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm
import glob
from typing import Optional, Dict
from torch import Tensor, nn
from torchvision.transforms import functional as F_vision
from PIL import Image

from common import SUPPORTED_IMAGE_EXTNS
from options.opts import get_bokeh_eval_arguments
from cvnets import get_model
from cvnets.models.detection.ssd import DetectionPredTuple
from data import create_eval_loader
from data.datasets.detection.coco_base import COCODetection
from utils.tensor_utils import to_numpy, image_size_from_opts
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master
from utils import logger
from engine.utils import print_summary
from utils.download_utils import get_local_path

from torchvision.utils import save_image
from engine.bokeh_utils.msssim import msssim
from engine.bokeh_utils.psnr import compute_psnr

import time

def predict_and_save(
    opts,
    input_tensor: Tensor,
    model: nn.Module,
    input_np: Optional[np.ndarray] = None,
    device: Optional = torch.device("cpu"),
    is_coco_evaluation: Optional[bool] = False,
    file_name: Optional[str] = None,
    output_stride: Optional[int] = 32,
    orig_h: Optional[int] = None,
    orig_w: Optional[int] = None,
    *args,
    **kwargs
):
    mixed_precision_training = getattr(opts, "common.mixed_precision", False)

    '''if input_np is None and not is_coco_evaluation:
        input_np = to_numpy(input_tensor).squeeze(  # convert to numpy
            0
        )  # remove batch dimension'''

    curr_height, curr_width = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_height // output_stride) * output_stride
    new_w = (curr_width // output_stride) * output_stride

    if new_h != curr_height or new_w != curr_width:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(
            input=input_tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    # move data to device
    input_tensor = input_tensor.to(device)

    with autocast(enabled=mixed_precision_training):
        # prediction
        prediction: DetectionPredTuple = model.forward(input_tensor, is_scaling=False)

    # convert tensors to numpy
    # pred = prediction.cpu().numpy()

    if orig_w is None:
        assert orig_h is None
        orig_h, orig_w = input_np.shape[:2]
    elif orig_h is None:
        assert orig_w is None
        orig_h, orig_w = input_np.shape[:2]
    assert orig_h is not None and orig_w is not None

    bokeh_res_file_name = getattr(opts, "common.exp_loc", None)
    if bokeh_res_file_name is None:
        logger.error('bokeh result save folder is None!')
    if not os.path.isdir(bokeh_res_file_name):
        os.makedirs(bokeh_res_file_name, exist_ok=True)
    if file_name is not None:
        file_name = str(file_name) + ".jpg"
        bokeh_res_file_name = "{}/{}".format(bokeh_res_file_name, file_name)
    # print(bokeh_res_file_name)
    save_image(prediction, bokeh_res_file_name)

    return prediction


def _get_batch_size(x):
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    elif isinstance(x, Dict):
        return x["image"].shape[0]


def predict_labeled_dataset(opts, **kwargs):
    device = getattr(opts, "dev.device", torch.device("cpu"))

    # set-up data loaders
    val_loader = create_eval_loader(opts)

    # set-up the model
    model = get_model(opts)
    model.eval()
    model = model.to(device=device)
    print_summary(opts=opts, model=model)

    if model.training:
        logger.warning("Model is in training mode. Switching to evaluation mode")
        model.eval()

    starttime = time.time()
    msLoss = 0
    psnrLoss = 0
    with torch.no_grad():
        predictions = []
        for img_idx, batch in tqdm(enumerate(val_loader)):
            input_img, target_label = batch["image"], batch["label"]

            # input_img = input_img["image"]

            batch_size = _get_batch_size(input_img)
            assert (
                batch_size == 1
            ), "We recommend to run bokeh evaluation with a batch size of 1"

            orig_w = target_label.shape[3]
            orig_h = target_label.shape[2]

            bokeh = predict_and_save(
                opts=opts,
                input_tensor=input_img,
                model=model,
                device=device,
                is_coco_evaluation=True,
                orig_w=orig_w,
                orig_h=orig_h,
                file_name = img_idx+4400
            )

    endtime = time.time()
    print('runtime per images: ', (endtime-starttime)/len(val_loader))

def read_and_process_image(opts, image_fname: str, *args, **kwargs):
    input_img = Image.open(image_fname).convert("RGB")
    input_np = np.array(input_img)
    orig_w, orig_h = input_img.size

    # Resize the image to the resolution that detector supports
    res_h, res_w = image_size_from_opts(opts)
    input_img = F_vision.resize(
        input_img,
        size=[res_h, res_w],
        interpolation=F_vision.InterpolationMode.BILINEAR,
    )
    input_tensor = F_vision.pil_to_tensor(input_img)
    input_tensor = input_tensor.float().div(255.0).unsqueeze(0)
    return input_tensor, input_np, orig_h, orig_w


def predict_image(opts, image_fname, **kwargs):
    image_fname = get_local_path(opts, image_fname)
    if not os.path.isfile(image_fname):
        logger.error("Image file does not exist at: {}".format(image_fname))

    input_tensor, input_imp_copy, orig_h, orig_w = read_and_process_image(
        opts, image_fname=image_fname
    )

    image_fname = image_fname.split(os.sep)[-1]

    device = getattr(opts, "dev.device", torch.device("cpu"))
    # set-up the model
    model = get_model(opts)
    model.eval()
    model = model.to(device=device)
    print_summary(opts=opts, model=model)

    if model.training:
        logger.warning("Model is in training mode. Switching to evaluation mode")
        model.eval()

    with torch.no_grad():
        predict_and_save(
            opts=opts,
            input_tensor=input_tensor,
            input_np=input_imp_copy,
            file_name=image_fname,
            model=model,
            device=device,
            orig_h=orig_h,
            orig_w=orig_w,
        )


def predict_images_in_folder(opts, **kwargs):
    img_folder_path = getattr(opts, "evaluation.bokeh.path", None)
    if img_folder_path is None:
        logger.error(
            "Image folder is not passed. Please use --evaluation.bokeh.path as an argument to pass the location of image folder".format(
                img_folder_path
            )
        )
    elif not os.path.isdir(img_folder_path):
        logger.error(
            "Image folder does not exist at: {}. Please check".format(img_folder_path)
        )

    img_files = []
    for e in SUPPORTED_IMAGE_EXTNS:
        img_files_with_extn = glob.glob("{}/*{}".format(img_folder_path, e))
        if len(img_files_with_extn) > 0 and isinstance(img_files_with_extn, list):
            img_files.extend(img_files_with_extn)

    if len(img_files) == 0:
        logger.error(
            "Number of image files found at {}: {}".format(
                img_folder_path, len(img_files)
            )
        )

    logger.log(
        "Number of image files found at {}: {}".format(img_folder_path, len(img_files))
    )

    device = getattr(opts, "dev.device", torch.device("cpu"))
    # set-up the model
    model = get_model(opts)
    model.eval()
    model = model.to(device=device)
    print_summary(opts=opts, model=model)

    if model.training:
        logger.warning("Model is in training mode. Switching to evaluation mode")
        model.eval()

    with torch.no_grad():
        for img_idx, image_fname in enumerate(img_files):
            input_tensor, input_np, orig_h, orig_w = read_and_process_image(
                opts=opts, image_fname=image_fname
            )

            image_fname = image_fname.split(os.sep)[-1]

            predict_and_save(
                opts=opts,
                input_tensor=input_tensor,
                input_np=input_np,
                file_name=image_fname,
                model=model,
                device=device,
                orig_h=orig_h,
                orig_w=orig_w,
            )


def main_bokeh_evaluation(**kwargs):
    opts = get_bokeh_eval_arguments()

    dataset_name = getattr(opts, "dataset.name", "ebb")

    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error("--rank should be >=0. Got {}".format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    logger.log("Results (if any) will be stored here: {}".format(exp_dir))

    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    if num_gpus < 2:
        cls_norm_type = getattr(opts, "model.normalization.name", "batch_norm_2d")
        if cls_norm_type.find("sync") > -1:
            # replace sync_batch_norm with standard batch norm on PU
            setattr(
                opts, "model.normalization.name", cls_norm_type.replace("sync_", "")
            )
            setattr(
                opts,
                "model.classification.normalization.name",
                cls_norm_type.replace("sync_", ""),
            )

    # we disable the DDP setting for evaluation tasks
    setattr(opts, "ddp.use_distributed", False)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    if dataset_workers == -1:
        setattr(opts, "dataset.workers", n_cpus)

    # We are not performing any operation like resizing and cropping on images
    # Because image dimensions are different, we process 1 sample at a time.
    setattr(opts, "dataset.train_batch_size0", 1)
    setattr(opts, "dataset.val_batch_size0", 1)
    setattr(opts, "dev.device_id", None)

    eval_mode = getattr(opts, "evaluation.bokeh.mode", "validation_set")

    if eval_mode == "single_image":
        num_classes = getattr(opts, "model.bokeh.n_classes", 81)
        assert num_classes is not None

        # test a single image
        img_f_name = getattr(opts, "evaluation.bokeh.path", None)
        predict_image(opts, img_f_name, **kwargs)
    elif eval_mode == "image_folder":
        num_seg_classes = getattr(opts, "model.bokeh.n_classes", 81)
        assert num_seg_classes is not None

        # test all images in a folder
        predict_images_in_folder(opts=opts, **kwargs)
    elif eval_mode == "validation_set":
        # evaluate and compute stats for labeled image dataset
        # This is useful for generating results for validation set and compute quantitative results
        predict_labeled_dataset(opts=opts, **kwargs)
    else:
        logger.error(
            "Supported modes are single_image, image_folder, and validation_set. Got: {}".format(
                eval_mode
            )
        )


if __name__ == "__main__":
    main_bokeh_evaluation()
