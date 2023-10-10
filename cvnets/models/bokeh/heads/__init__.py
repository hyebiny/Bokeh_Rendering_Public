#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
from utils import logger
from typing import Dict
import argparse

from .base_bok_head import BaseBokHead


BOK_HEAD_REGISTRY = {}


def register_bokeh_head(name):
    def register_model_class(cls):
        if name in BOK_HEAD_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        if not issubclass(cls, BaseBokHead):
            raise ValueError(
                "Model ({}: {}) must extend BaseBokHead".format(name, cls.__name__)
            )

        BOK_HEAD_REGISTRY[name] = cls
        return cls

    return register_model_class


def build_bokeh_head(opts, enc_conf: Dict, use_l5_exp: bool = False, features=128):
    '''bok_model_name = getattr(opts, "model.bokeh.bok_head", "bokeh")
    bok_head = None
    print('---------------hyebin in models/bokeh/heads/__init__.py')
    print(bok_model_name)
    print(BOK_HEAD_REGISTRY)
    if bok_model_name in BOK_HEAD_REGISTRY:
        bok_head = BOK_HEAD_REGISTRY[bok_model_name](
            opts=opts, enc_conf=enc_conf
        )
    else:
        supported_heads = list(BOK_HEAD_REGISTRY.keys())
        supp_model_str = "Supported bokeh heads are:"
        for i, m_name in enumerate(supported_heads):
            supp_model_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        logger.error(supp_model_str)

    return bok_head'''

    return BaseBokHead(opts, enc_conf=enc_conf, features=features)


def arguments_bokeh_head(parser: argparse.ArgumentParser):
    # add bokeh head specific arguments
    # parser = BaseBokHead.add_arguments(parser=parser)
    # for k, v in BOK_HEAD_REGISTRY.items():
        # parser = v.add_arguments(parser=parser)

    return parser


# automatically import the models
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "cvnets.models.bokeh.heads." + model_name
        )
