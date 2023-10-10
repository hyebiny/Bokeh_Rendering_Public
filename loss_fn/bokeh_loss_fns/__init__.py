#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib

SUPPORTED_BOK_LOSS_FNS = []


def register_bokeh_loss_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_BOK_LOSS_FNS:
            raise ValueError(
                "Cannot register duplicate bokeh_loss_fns loss function ({})".format(name)
            )
        SUPPORTED_BOK_LOSS_FNS.append(name)
        return fn

    return register_fn


# automatically import different loss functions
loss_fn_dir = os.path.dirname(__file__)
for file in os.listdir(loss_fn_dir):
    path = os.path.join(loss_fn_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("loss_fn.bokeh_loss_fns." + model_name)


from loss_fn.bokeh_loss_fns.msssim import MSSSIM
from loss_fn.bokeh_loss_fns.ssim import SSIM
from loss_fn.bokeh_loss_fns.l1msssim import MS_SSIM_L1_LOSS
