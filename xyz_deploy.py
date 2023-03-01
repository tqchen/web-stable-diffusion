from typing import Dict, List

import argparse
import time

from transformers import CLIPTokenizer
from diffusers import PNDMScheduler, EulerDiscreteScheduler

import numpy as np
import tvm
import torch
from tvm import relax, rpc
import json

from tvm.contrib import tvmjs
from utils import load_params, wrapper, remote_wrapper
from xyz_pipeline import TVMSDPipeline


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--lib-path", type=str, default="module.so")
    args.add_argument("--mode", type=str, default="webgpu")
    args.add_argument("--ndarray-cache-path", type=str,
                      default="../tvm/web/.ndarray_cache/sd-metal-v1-5")
    parsed = args.parse_args()
    return parsed


class MetalWrapper:
    def __init__(self, vm, param_dict, tvm_device, torch_device):
        self.clip = wrapper(vm["clip"], param_dict["clip"], tvm_device, torch_device)
        self.vae = wrapper(vm["vae"], param_dict["vae"], tvm_device, torch_device)
        self.unet = wrapper(vm["unet"], param_dict["unet"], tvm_device, torch_device)

class WebGPUWrapper:
    def __init__(self, wasm_path, torch_device):
        proxy_host = "127.0.0.1"
        proxy_port = 9090
        meta_data = json.load(open(
            "../tvm/web/.ndarray_cache/sd-metal-v1-5/ndarray-cache.json", "r"))["meta_data"]

        wasm_binary = open(wasm_path, "rb").read()
        remote = rpc.connect(
            proxy_host,
            proxy_port,
            key="wasm",
            session_constructor_args=["rpc.WasmSession", wasm_binary],
        )
        dev = remote.webgpu(0)
        vm = relax.VirtualMachine(remote.system_lib(), device=dev)

        self.clip = remote_wrapper(remote, vm, "clip", meta_data["clip_param_size"], dev, torch_device)
        self.unet = remote_wrapper(remote, vm, "unet", meta_data["unet_param_size"], dev, torch_device)
        self.vae = remote_wrapper(remote, vm, "vae", meta_data["vae_param_size"], dev, torch_device)
        print("Finish initialization")


def deploy(lib_path, cache_path, mode):
    torch_device=torch.device("mps")
    def get_metel_wrapper():
        dev = tvm.metal()
        param_dict = load_params(cache_path, dev)

        ex = tvm.runtime.load_module(lib_path)
        vm = relax.VirtualMachine(ex, dev)
        return  MetalWrapper(vm, param_dict, dev, torch_device)

    wgpu_wrapper = WebGPUWrapper("build/vae.wasm", torch_device)
    metal_wrapper = get_metel_wrapper()

    wrapper = wgpu_wrapper
    # wrapper.clip = metal_wrapper.clip
    #wrapper.unet = metal_wrapper.unet
    # wrapper.vae = metal_wrapper.vae

    pipe = TVMSDPipeline(
        wrapper,
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
        scheduler=PNDMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        ),
        torch_device=torch_device,
    )

    prompt = "A photo of an astronaut riding a horse on mars"


    start = time.time()
    image = pipe([prompt])[0]
    end = time.time()

    image.save("example.png")
    print(f"Time elapsed: {end - start} seconds")


if __name__ == "__main__":
    ARGS = _parse_args()
    deploy(ARGS.lib_path, ARGS.ndarray_cache_path, ARGS.mode)