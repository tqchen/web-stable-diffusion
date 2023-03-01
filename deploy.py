import argparse
import time

from transformers import CLIPTokenizer
from scheduler import PNDMScheduler

import tvm
import torch
from tvm import relax

from pipeline import TVMSDPipeline
from utils import load_constant_from_file


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--const-params-dir", type=str)
    args.add_argument("--lib-path", type=str)
    parsed = args.parse_args()
    return parsed


def deploy_to_pipeline(const_params_dir, lib_path):
    dev = tvm.metal()
    const_params_dict = load_constant_from_file(dir=const_params_dir, dev=dev)

    ex = tvm.runtime.load_module(lib_path)
    vm = relax.VirtualMachine(ex, dev)

    pipe = TVMSDPipeline(
        vm=vm,
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
        scheduler=PNDMScheduler(dev),
        tvm_device=dev,
        torch_device=torch.device("mps"),
        param_dict=const_params_dict,
    )

    prompt = "A photo of an astronaut riding a horse on mars"

    start = time.time()
    image = pipe([prompt])[0]
    end = time.time()

    image.save("example.png")
    print(f"Time elapsed: {end - start} seconds")


if __name__ == "__main__":
    ARGS = _parse_args()
    deploy_to_pipeline(ARGS.const_params_dir, ARGS.lib_path)
