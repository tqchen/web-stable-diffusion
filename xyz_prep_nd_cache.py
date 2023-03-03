from typing import Dict, List

import argparse
import time

import numpy as np
import tvm
import torch
from tvm import relax
from tvm.contrib import tvmjs
from utils import load_constant_from_file

def prepare_params():
    const_dict = load_constant_from_file("const_params", tvm.cpu(), deploy=True)
    meta_data = {}
    param_dict = {}
    for model in ["unet", "vae", "clip"]:
        meta_data[f"{model}ParamSize"] = len(const_dict[model])
        for i, nd in enumerate(const_dict[model]):
            param_dict[f"{model}_{i}"] = nd
    tvmjs.dump_ndarray_cache(param_dict,
                             "../tvm/web/.ndarray_cache/sd-webgpu-v1-5",
                             meta_data=meta_data)

prepare_params()
