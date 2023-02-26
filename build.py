import os

import argparse
import numpy as np
import pickle
from typing import Dict, List

import tvm
import tvm.testing
from tvm import meta_schedule as ms, relax
import tvm.tir.tensor_intrin


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--target", type=str)
    args.add_argument("--target-host", type=str, default="llvm")
    args.add_argument("--log-dir", type=str)
    args.add_argument("--output", type=str)
    args.add_argument("--llvm-output", type=str)

    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target, host=parsed.target_host)
    return parsed


def build(
    target,
    log_dir,
    output,
    llvm_output,
):
    model_names = ["clip", "vae", "unet"]

    # Build the main functions.
    if not os.path.exists(output):
        pkl_file = open("irmodule.pkl", "rb")
        relax_mod = pickle.load(pkl_file)
        pkl_file.close()

        relax_mod = relax.transform.RemoveUnusedFunctions(model_names)(relax_mod)
        db = ms.database.create(work_dir=log_dir)
        with target, db, tvm.transform.PassContext(opt_level=3):
            relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)

        ex = relax.build(relax_mod, target)
        ex.mod.export_library(output)

    # Build the param transformation functions.
    if not os.path.exists(llvm_output):
        pkl_file = open("irmodule.pkl", "rb")
        relax_mod = pickle.load(pkl_file)
        pkl_file.close()

        relax_mod = relax.transform.RemoveUnusedFunctions(
            [name + "_transform_params" for name in model_names]
        )(relax_mod)

        ex = relax.build(relax_mod, target="llvm")
        ex.mod.export_library(llvm_output)


if __name__ == "__main__":
    ARGS = _parse_args()
    build(
        ARGS.target,
        ARGS.log_dir,
        ARGS.output,
        ARGS.llvm_output,
    )
