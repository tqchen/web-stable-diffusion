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
import time
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
wasm_path = "build/vae.wasm"

def build_webgpu(skip_build):
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

def main_show_image():
    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )
    latents = torch.load("intermediate/vae_image_metal.pt")
    rawdata = latents.cpu().numpy()
    data = tvm.nd.array(rawdata, remote.webgpu(0))
    print(data.shape)
    image = numpy_to_pil(rawdata)
    image[0].save("build/vae_pair_show.png")
    remote.get_function("showImage")(data)
    input()

    latents = torch.load("intermediate/vae_image_webgpu.pt")
    rawdata = latents.cpu().numpy()
    rawdata = np.zeros_like(rawdata)
    rawdata[:, :, :, 2] = 1
    print(rawdata)
    data = tvm.nd.array(rawdata, remote.webgpu(0))
    remote.get_function("clearImage")(data)
    print("finish")

def main_run_vae():
    latents = torch.load("intermediate/latents.pt")
    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )
    rawdata = latents.cpu().numpy()
    data = tvm.nd.array(rawdata, remote.webgpu(0))

    remote.get_function("runVAEStage")(data)

def main_run_unet():
    latents = torch.load("intermediate/unet_input_0.pt")
    embedding = torch.load("intermediate/clip_output.pt")

    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )
    print(embedding.shape)
    latents = tvm.nd.array(latents.cpu().numpy(), remote.webgpu(0))
    embedding = tvm.nd.array(embedding.cpu().numpy(), remote.webgpu(0))
    tstart = time.time()
    remote.get_function("runUNetStage")(latents, embedding, 50, 100)
    tend = time.time()

    print(f"Time ={tend - tstart}")

    input()
    print("second attempt")
    tstart = time.time()
    remote.get_function("runUNetStage")(latents, embedding, 50, 100)
    tend = time.time()
    print(f"Time ={tend - tstart}")


def main_run_clip():
    input_ids = torch.load("intermediate/clip_input.pt")

    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )
    latents = torch.load("intermediate/unet_input_0.pt")
    latents = tvm.nd.array(latents.cpu().numpy(), remote.webgpu(0))
    input_ids = input_ids.cpu().numpy().astype("int32")
    prompt = "A photo of an astronaut flying over the moon"

    print(input_ids)
    input_ids = tvm.nd.array(input_ids, remote.webgpu(0))

    tstart = time.time()
    remote.get_function("generate")(prompt, 2)
    tend = time.time()
    input()
    tstart = time.time()
    remote.get_function("generate")(prompt, 2)
    tend = time.time()


def main_tint():
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


main_metal()
