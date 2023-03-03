import os

import argparse
import numpy as np
import pickle
from typing import Dict, List
from tvm import te

import tvm
import tvm.testing
from tvm.script import relax as R
from tvm import meta_schedule as ms, relax
import tvm.tir.tensor_intrin


def preproc(log_dir):
    model_names = ["clip", "vae", "unet"]
    scheduler_func_names = [f"scheduler_step_{i}" for i in range(5)]
    target = tvm.target.Target("apple/m2-gpu", host="llvm")
    # Build the main functions.

    pkl_file = open("irmodule.pkl", "rb")
    relax_mod = pickle.load(pkl_file)
    pkl_file.close()

    relax_mod = relax.transform.RemoveUnusedFunctions(
        model_names + scheduler_func_names
    )(relax_mod)
    db = ms.database.create(work_dir=log_dir)
    with target, db, tvm.transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)

    return relax_mod


def get_to_rgba():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([1, 512, 512, 3], "float32"))
    def te_to_rgba(A):
        def scale_clamp(v):
            v = te.round(v * 255.0)
            v = te.max(v, 0.0)
            v = te.min(v, 255.0)
            v = v.astype("uint32")
            return v
        def fcompute(y, x):
            b = scale_clamp(A[0][y][x][0])
            g = scale_clamp(A[0][y][x][1])
            r = scale_clamp(A[0][y][x][2])
            return b | (g << 8) | (r << 16) | tvm.tir.const(255 << 24, "uint32")
        return te.compute((512, 512), fcompute, name="B")


    with bb.function("image_to_rgba", [x]):
        with bb.dataflow():
            gv = bb.emit_output(bb.call_te(te_to_rgba, x))
        bb.emit_func_output(gv)
    mod = bb.get()

    mod.show()
    sch = tvm.tir.Schedule(mod)
    # manually transform loop
    sch.work_on("te_to_rgba")
    loops = sch.get_loops(block=sch.get_block("B"))
    i = sch.fuse(*loops)
    i0, i1 = sch.split(i, [None, 128])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    return sch.mod

def main():
    mod = get_to_rgba()
    print(mod.script(show_meta=True))

main()
