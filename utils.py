from typing import Dict, List

import os

import numpy as np
import pickle

import tvm
from tvm import relax


def load_constant_from_file(dir: str, dev) -> Dict[str, List[tvm.nd.NDArray]]:
    const_dict: Dict[str, List[np.ndarray]] = dict()
    for model_name in ["clip", "vae", "unet"]:
        n_file = len(os.listdir(f"{dir}/{model_name}/"))
        constants = list()
        for i in range(n_file):
            file = open(f"{dir}/{model_name}/{model_name}_{i}.pkl", "rb")
            const = pickle.load(file)
            file.close()
            constants.append(tvm.nd.array(const, device=dev))
        const_dict[model_name] = constants
    return const_dict


def compute_params(
    vm: relax.VirtualMachine, const_dict: Dict[str, List[np.ndarray]]
) -> Dict[str, List[np.ndarray]]:
    param_dict = dict()
    for name, constants in const_dict.items():
        params = vm[name + "_transform_params"](constants)
        param_dict[name] = [param.numpy() for param in params]
    return param_dict