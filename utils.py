from typing import Dict, List

import os

import numpy as np
import pickle

import tvm
from tvm import relax
import tvm
import torch
from PIL import Image
from tvm.contrib import tvmjs


def load_constant_from_file(dir: str, dev, deploy: bool) -> Dict[str, List[tvm.nd.NDArray]]:
    const_dict: Dict[str, List[np.ndarray]] = dict()
    for model_name in ["clip", "vae", "unet"]:
        n_file = len(os.listdir(f"{dir}/{model_name}/"))
        constants = list()
        for i in range(n_file):
            if not deploy and model_name == "clip":
                if i == 0:
                    filename = f"{dir}/clip/uncond_embeddings.pkl"
                else:
                    filename = f"{dir}/{model_name}/{model_name}_{i - 1}.pkl"
            else:
                filename = f"{dir}/{model_name}/{model_name}_{i}.pkl"
            file = open(filename, "rb")
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

def load_params(cache_path, device):
    pdict = {}
    params, meta = tvmjs.load_ndarray_cache(cache_path, device)
    for model in ["vae", "unet", "clip"]:
        plist = []
        size = meta[f"{model}_param_size"]
        for i in range(size):
            plist.append(params[f"{model}_{i}"])
        pdict[model] = plist
    return pdict


def _tvm_to_torch(data, torch_device):
    if isinstance(data, (tvm.ir.Array, list, tuple)):
        return [_tvm_to_torch(i, torch_device) for i in data]
    return torch.from_numpy(data.numpy()).to(torch_device)


def torch_wrapper(vm, model, params, tvm_device, torch_device, time_eval=False):
    time_eval_result = []

    def wrapped_f(*args):
        new_args = []
        for arg in args:
            np_arg = arg.cpu().numpy()
            if np_arg.shape == ():
                np_arg = np_arg.reshape((1,))
            new_args.append(tvm.nd.array(np_arg, tvm_device))

        if time_eval and len(time_eval_result) == 0:
            res = vm.time_evaluator(model, tvm_device)(*new_args, params)
            time_eval_result.append(res)
            print(f"Local[{model}] on {tvm_device}, time evaluator {model}, {res}")
        return _tvm_to_torch(vm[model](*new_args, params), torch_device)
    return wrapped_f


def remote_torch_wrapper(remote, vm, model, nparams, tvm_device, torch_device, time_eval=False):
    pfunc_from_cache = remote.get_function("tvmjs.param_module_from_cache")
    pfunc = pfunc_from_cache(model, nparams)
    time_eval_result = []

    def wrapped_f(*args):
        new_args = []
        for arg in args:
            np_arg = arg.cpu().numpy()
            if np_arg.shape == ():
                np_arg = np_arg.reshape((1,))
            new_args.append(tvm.nd.array(np_arg, tvm_device))
        vm.module["set_input_with_param_module"](model, *new_args, pfunc)
        vm.invoke_stateful(model)
        if time_eval and len(time_eval_result) == 0:
            res = vm.time_evaluator("invoke_stateful", tvm_device, number=1)(model)
            time_eval_result.append(res)
            print(f"Remote[{model}] on {tvm_device}, {res}")

        results = vm.get_outputs(model)
        return _tvm_to_torch(results, torch_device)
    return wrapped_f


def tvm_wrapper(vm, model, params, tvm_device, time_eval=False):
    time_eval_result = []

    def wrapped_f(*args):
        new_args = []
        for arg in args:
            if arg.device != tvm_device:
                arg = arg.copyto(tvm.cpu()).copyto(tvm_device)
            new_args.append(arg)

        if time_eval and len(time_eval_result) == 0:
            res = vm.time_evaluator(model, tvm_device)(*new_args, params)
            time_eval_result.append(res)
            print(f"Local[{model}] on {tvm_device}, time evaluator {model}, {res}")
        return vm[model](*new_args, params)
    return wrapped_f


def remote_tvm_wrapper(remote, vm, model, nparams, tvm_device, time_eval=False):
    pfunc_from_cache = remote.get_function("tvmjs.param_module_from_cache")
    pfunc = pfunc_from_cache(model, nparams)
    time_eval_result = []

    def wrapped_f(*args):
        new_args = []
        for arg in args:
            if arg.device != tvm_device:
                arg = arg.copyto(tvm.cpu()).copyto(tvm_device)
            new_args.append(arg)
        vm.module["set_input_with_param_module"](model, *new_args, pfunc)
        vm.invoke_stateful(model)
        if time_eval and len(time_eval_result) == 0:
            res = vm.time_evaluator("invoke_stateful", tvm_device, number=1)(model)
            time_eval_result.append(res)
            print(f"Remote[{model}] on {tvm_device}, {res}")

        return vm.get_outputs(model)

    return wrapped_f


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [
            Image.fromarray(image.squeeze(), mode="L") for image in images
        ]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
