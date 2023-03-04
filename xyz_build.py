import os

import argparse
import numpy as np

import tvm
from tvm.contrib import tvmjs
import tvm.testing
from tvm import meta_schedule as ms, relax
import webgpu_module

def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--target", type=str, default="apple/m2-gpu")
    args.add_argument("--output", type=str, default="module.so")

    parsed = args.parse_args()
    if parsed.target == "webgpu":
        parsed.target = tvm.target.Target("webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm")
        parsed.output = "build/stable_diffusion.wasm"
    else:
        parsed.target = tvm.target.Target(parsed.target, host="llvm")
    return parsed


def build(
    target,
    output,
):
    # Build the main functions.
    relax_mod = webgpu_module.Module
    ex = relax.build(relax_mod, target)
    ex.mod.export_library(output)
    print(f"Finish exporting to {output}")

if __name__ == "__main__":
    ARGS = _parse_args()
    build(
        ARGS.target,
        ARGS.output,
    )
