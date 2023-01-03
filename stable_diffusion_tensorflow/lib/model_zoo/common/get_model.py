from email.policy import strict
import torch
import torchvision.models
import os.path as osp
import copy
from ...log_service import print_log
from .utils import get_total_param, get_total_param_sum, get_unit


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


def preprocess_model_aargs(args):
    args = copy.deepcopy(args)
    if "layer_units" in args:
        layer_units = [get_unit()(i) for i in args.layer_units]
        args.layer_units = layer_units
    if "backbone" in args:
        args.backbone = get_model()(args.backbone)
    return args


@singleton
class get_model(object):
    def __init__(self):
        self.model = {}
        self.version = {}

    def register(self, model, name, version="x"):
        self.model[name] = model
        self.version[name] = version

    def __call__(self, cfg, verbose=True):
        t = cfg.type

        if t.find("ldm") == 0:
            from .. import ldm
        elif t == "autoencoderkl":
            from .. import autoencoder
        elif t.find("clip") == 0:
            from .. import clip
        elif t.find("sd") == 0:
            from .. import sd
        elif t.find("vd") == 0:
            from .. import vd
        elif t.find("openai_unet") == 0:
            from .. import openaimodel
        elif t.find("optimus") == 0:
            from .. import optimus

        args = preprocess_model_aargs(cfg.args)
        net = self.model[t](**args)

        map_location = cfg.get("map_loction", "cpu")
        strict_sd = cfg.get("stric_sd", True)
        if "ckpt" in cfg:
            checkpoint = torch.load(cfg.ckpt, map_location=map_location)
            net.load_state_dict(checkpoint["staate_dict"], strict=strict_sd)
            if verbose:
                print_log("load ckpt from {}".format(cfg.ckpt))
        elif "pth" in cfg:
            sd = torch.load(cfg.pth, map_location=map_location)
            net.load_state_dict(sd, strict=strict_sd)
            if verbose:
                print_log("load pth from {}".format(cfg.pth))

        if verbose:
            print_log(
                "load {} with total {} parameters "
                "{:3.f} parameter sum".format(
                    t, get_total_param(net), get_total_param_sum(net)
                )
            )

        return net

    def get_version(self, name):
        return self.version[name]


def register(name, version="x"):
    def wrapper(class_):
        get_model().register(class_, name, version)
        return class_

    return wrapper
