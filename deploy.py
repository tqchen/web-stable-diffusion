from typing import Dict, List

import argparse
import time

from transformers import CLIPTokenizer
from diffusers import PNDMScheduler

import numpy as np
import tvm
import torch
from tvm import relax

from pipeline import TVMSDPipeline


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--lib-path", type=str)
    args.add_argument("--lib-llvm-path", type=str)
    parsed = args.parse_args()
    return parsed


def load_constant_from_file(weight_dir: str) -> Dict[str, List[np.ndarray]]:
    import os
    import pickle

    const_dict: Dict[str, List[np.ndarray]] = dict()
    for model_name in ["clip", "vae", "unet"]:
        n_file = len(os.listdir(f"{weight_dir}/{model_name}/"))
        constants = list()
        for i in range(n_file):
            file = open(f"{weight_dir}/{model_name}/{model_name}_{i}.pkl", "rb")
            const = pickle.load(file)
            file.close()
            constants.append(const)
        const_dict[model_name] = constants
    return const_dict


def compute_params(
    vm: relax.VirtualMachine, const_dict: Dict[str, List[np.ndarray]], dev_cpu, dev
):
    param_dict = dict()
    for name, constants in const_dict.items():
        inputs = [tvm.nd.array(const, dev_cpu) for const in constants]
        params = vm[name + "_transform_params"](inputs)
        param_dict[name] = [tvm.nd.array(param.numpy(), dev) for param in params]
    return param_dict


def deploy_to_pipeline(lib_path, lib_llvm_path):
    ex_llvm = tvm.runtime.load_module(lib_llvm_path)
    vm_llvm = relax.VirtualMachine(ex_llvm, tvm.cpu())

    dev = tvm.metal()
    const_dict = load_constant_from_file(weight_dir="weight_pkl")
    param_dict = compute_params(vm_llvm, const_dict, tvm.cpu(), dev)
    del const_dict

    ex = tvm.runtime.load_module(lib_path)
    vm = relax.VirtualMachine(ex, dev)

    pipe = TVMSDPipeline(
        vm=vm,
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
        scheduler=PNDMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        ),
        tvm_device=dev,
        torch_device=torch.device("mps"),
        param_dict=param_dict,
    )

    prompt = "A photo of an astronaut riding a horse on mars"

    start = time.time()
    image = pipe([prompt])[0]
    end = time.time()

    image.save("example.png")
    print(f"Time elapsed: {end - start} seconds")


if __name__ == "__main__":
    ARGS = _parse_args()
    deploy_to_pipeline(ARGS.lib_path, ARGS.lib_llvm_path)
