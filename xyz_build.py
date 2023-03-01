import os

import argparse
import numpy as np

import tvm
import tvm.testing
from tvm import meta_schedule as ms, relax
import tvm.tir.tensor_intrin
import webgpu_module

def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--target", type=str, default="apple/m2-gpu")
    args.add_argument("--output", type=str, default="module.so")

    parsed = args.parse_args()
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


if __name__ == "__main__":
    ARGS = _parse_args()
    build(
        ARGS.target,
        ARGS.output,
    )
