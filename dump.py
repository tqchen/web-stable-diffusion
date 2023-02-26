import os

import argparse
import numpy as np
import pickle
from typing import Dict, List

import tvm
import tvm.testing
from tvm import meta_schedule as ms, relax
import tvm.tir.tensor_intrin


def preproc(log_dir):
    model_names = ["clip", "vae", "unet"]
    target = tvm.target.Target("apple/m2-gpu", host="llvm")
    # Build the main functions.

    pkl_file = open("irmodule.pkl", "rb")
    relax_mod = pickle.load(pkl_file)
    pkl_file.close()

    relax_mod = relax.transform.RemoveUnusedFunctions(model_names)(relax_mod)
    db = ms.database.create(work_dir=log_dir)
    with target, db, tvm.transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)

    return relax_mod


def main():
    mod = preproc("log_db/webgpu_0228")
    print(mod.script(show_meta=True))

main()
