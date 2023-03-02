import os

import argparse
import numpy as np
import pickle
from typing import Dict, List
import torch
import tvm
import tvm.testing
from tvm import meta_schedule as ms, relax
from tvm import rpc

import json
from tvm.contrib import tvmjs
from utils import numpy_to_pil, torch_wrapper as wrapper, load_params, remote_torch_wrapper as remote_wrapper

def build_metal():
    import webgpu_module
    relax_mod = webgpu_module.Module
    target = tvm.target.Target("apple/m2-gpu", host="llvm")
    ex = relax.build(relax_mod, target)
    vm = relax.VirtualMachine(ex, tvm.metal())
    return vm


proxy_host = "127.0.0.1"
proxy_port = 9090

def build_webgpu(skip_build):
    wasm_path = "build/vae.wasm"
    nparams = int(json.load(open(
        "../tvm/web/.ndarray_cache/sd-webgpu-v1-5/ndarray-cache.json", "r"))["meta_data"]["vae_param_size"])
    print(f"nparams={nparams}")
    if not skip_build:
        import webgpu_module
        relax_mod = webgpu_module.Module
        target = tvm.target.Target("webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm")
        ex = relax.build(relax_mod, target)
        ex.export_library(wasm_path, tvmjs.create_tvmjs_wasm)
    print("finish export")

    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )
    dev = remote.webgpu(0)
    vm = relax.VirtualMachine(remote.system_lib(), device=dev)

    return remote_wrapper(remote, vm, "vae", nparams, remote.webgpu(0), torch.device("mps"), time_eval=True)


def main_webgpu():
    latents = torch.load("intermediate/latents.pt")
    print("finish load")
    vae = build_webgpu(False)
    print("finish build")
    image = vae(latents)
    torch.save(image, "intermediate/vae_image_webgpu.pt")
    print("finish exec")
    image = image.cpu().numpy()
    image = numpy_to_pil(image)
    image[0].save("build/vae_pair_webgpu.png")


def main_metal():
    dev = tvm.metal()
    param_dict = load_params("../tvm/web/.ndarray_cache/sd-webgpu-v1-5", dev)
    latents = torch.load("intermediate/latents.pt")
    print("finish load")
    vm = build_metal()
    print("finish build")
    vae = wrapper(vm, "vae", param_dict["vae"], tvm.metal(), torch.device("mps"), time_eval=True)
    image = vae(latents)
    torch.save(image, "intermediate/vae_image_metal.pt")
    print("finish exec")
    image = image.cpu().numpy()
    image = numpy_to_pil(image)
    image[0].save("build/vae_pair_metal.png")


main_webgpu()
