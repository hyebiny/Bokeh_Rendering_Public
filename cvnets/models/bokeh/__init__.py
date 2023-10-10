#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from .base_bok import BaseBokeh
import os
import importlib
import argparse

from utils.download_utils import get_local_path
from utils import logger
from utils.common_utils import check_frozen_norm_layer
from utils.ddp_utils import is_master, is_start_rank_node

from ...misc.common import load_pretrained_model
from ..classification import build_classification_model

BOK_MODEL_REGISTRY = {}


def register_bokeh_models(name):
    def register_model_class(bok):
        if name in BOK_MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        if not issubclass(bok, BaseBokeh):
            raise ValueError(
                "Model ({}: {}) must extend BaseBokeh".format(name, bok.__name__)
            )

        BOK_MODEL_REGISTRY[name] = bok
        return bok

    return register_model_class


def build_bokeh_model(opts, *args, **kwargs):
    model_name = getattr(opts, "model.bokeh.name", None) # "encoder_decoder")
    model = None
    is_master_node = is_master(opts)
    if model_name in BOK_MODEL_REGISTRY:
        encoder = build_classification_model(opts=opts)
        bok_act_fn = getattr(opts, "model.bokeh.activation.name", None)
        if bok_act_fn is not None:
            # Override the general activation arguments
            gen_act_fn = getattr(opts, "model.activation.name", "relu")
            gen_act_inplace = getattr(opts, "model.activation.inplace", False)
            gen_act_neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)

            setattr(opts, "model.activation.name", bok_act_fn)
            setattr(
                opts,
                "model.activation.inplace",
                getattr(opts, "model.bokeh.activation.inplace", False),
            )
            setattr(
                opts,
                "model.activation.neg_slope",
                getattr(opts, "model.bokeh.activation.neg_slope", 0.1),
            )

            model = BOK_MODEL_REGISTRY[model_name](opts, encoder) # *args, **kwargs)

            # Reset activation args
            setattr(opts, "model.activation.name", gen_act_fn)
            setattr(opts, "model.activation.inplace", gen_act_inplace)
            setattr(opts, "model.activation.neg_slope", gen_act_neg_slope)
        else:
            model = BOK_MODEL_REGISTRY[model_name](opts, encoder)
            # model = BOK_MODEL_REGISTRY[model_name](opts, *args, **kwargs)
    else:
        supported_models = list(BOK_MODEL_REGISTRY.keys())
        supp_model_str = "Supported models are:"
        for i, m_name in enumerate(supported_models):
            supp_model_str += "\n\t {}: {}".format(i, logger.color_text(m_name))

        if is_master_node:
            logger.error(supp_model_str)

    finetune_task = getattr(
        opts, "model.bokeh.finetune_pretrained_model", False
    )
    pretrained = getattr(opts, "model.bokeh.pretrained", None)
    if finetune_task:
        n_pretrained_classes = getattr(
            opts, "model.bokeh.n_pretrained_classes", None
        )
        n_classes = getattr(opts, "model.bokeh.n_classes", None)
        assert n_pretrained_classes is not None
        assert n_classes is not None

        # The model structure is the same as pre-trained model now
        model.update_classifier(opts, n_classes=n_pretrained_classes)

        # load the weights
        if pretrained is not None:
            pretrained = get_local_path(opts, path=pretrained)
            model = load_pretrained_model(
                model=model, wt_loc=pretrained, is_master_node=is_start_rank_node(opts)
            )

        # Now, re-initialize the bokeh layer
        model.update_classifier(opts, n_classes=n_classes)

    elif pretrained is not None:
        pretrained = get_local_path(opts, path=pretrained)
        model = load_pretrained_model(
            model=model, wt_loc=pretrained, is_master_node=is_start_rank_node(opts)
        )

    freeze_norm_layers = getattr(opts, "model.bokeh.freeze_batch_norm", False)
    if freeze_norm_layers:
        model.freeze_norm_layers()
        frozen_state, count_norm = check_frozen_norm_layer(model)
        if count_norm > 0 and frozen_state and is_master_node:
            logger.error(
                "Something is wrong while freezing normalization layers. Please check"
            )

        if is_master_node:
            logger.log("Normalization layers are frozen")

    return model


def std_bok_model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Bokeh arguments", description="Bokeh arguments"
    )
    group.add_argument(
        "--model.bokeh.classifier-dropout",
        type=float,
        default=0.0,
        help="Dropout rate in classifier",
    )

    group.add_argument(
        "--model.bokeh.name", type=str, default=None, help="Model name"
    )
    group.add_argument(
        "--model.bokeh.n-classes",
        type=int,
        default=1000,
        help="Number of classes in the dataset",
    )
    group.add_argument(
        "--model.bokeh.pretrained",
        type=str,
        default=None,
        help="Path of the pretrained backbone",
    )
    group.add_argument(
        "--model.bokeh.freeze-batch-norm",
        action="store_true",
        help="Freeze batch norm layers",
    )
    group.add_argument(
        "--model.bokeh.activation.name",
        default=None,
        type=str,
        help="Non-linear function name (e.g., relu)",
    )
    group.add_argument(
        "--model.bokeh.activation.inplace",
        action="store_true",
        help="Inplace non-linear functions",
    )
    group.add_argument(
        "--model.bokeh.activation.neg-slope",
        default=0.1,
        type=float,
        help="Negative slope in leaky relu",
    )

    group.add_argument(
        "--model.bokeh.finetune-pretrained-model",
        action="store_true",
        help="Finetune a pretrained model",
    )
    group.add_argument(
        "--model.bokeh.n-pretrained-classes",
        type=int,
        default=None,
        help="Number of pre-trained classes",
    )

    return parser


def common_bok_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Bokeh arguments", description="Bokeh arguments"
    )

    group.add_argument(
        "--model.bokeh.name", type=str, default=None, help="Model name"
    )
    group.add_argument(
        "--model.bokeh.pretrained",
        type=str,
        default=None,
        help="Path of the pretrained bokeh model. Useful for evaluation",
    )
    group.add_argument(
        "--model.bokeh.activation.name",
        default=None,
        type=str,
        help="Non-linear function type",
    )
    group.add_argument(
        "--model.bokeh.activation.inplace",
        action="store_true",
        help="Inplace non-linear functions",
    )
    group.add_argument(
        "--model.bokeh.activation.neg-slope",
        default=0.1,
        type=float,
        help="Negative slope in leaky relu",
    )
    group.add_argument(
        "--model.bokeh.freeze-batch-norm",
        action="store_true",
        help="Freeze batch norm layers",
    )
    group.add_argument(
        "--model.bokeh.use-level5-exp",
        action="store_true",
        help="Use output of Level 5 expansion layer in base feature extractor",
    )
    group.add_argument(
        "--model.bokeh.features",
        default=128,
        type=int,
        help="the number of features",
    )

    return parser

def arguments_bokeh(parser: argparse.ArgumentParser):
    parser = common_bok_args(parser)

    # add bokeh specific arguments
    for k, v in BOK_MODEL_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    from cvnets.models.bokeh.heads import arguments_bokeh_head
    parser = arguments_bokeh_head(parser)

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
        module = importlib.import_module("cvnets.models.bokeh." + model_name)
        ## by doing this command, code access to the all python files in bokeh folder
