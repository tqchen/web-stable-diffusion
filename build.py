import os

import argparse
import pickle

import tvm
import tvm.testing
from tvm import meta_schedule as ms, relax

from utils import load_constant_from_file, compute_params


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--target", type=str)
    args.add_argument("--target-host", type=str, default="llvm")
    args.add_argument("--log-dir", type=str)
    args.add_argument("--const-params-dir", type=str)
    args.add_argument("--output", type=str)

    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target, host=parsed.target_host)
    return parsed


def build(
    target,
    log_dir,
    const_params_dir,
    output,
):
    model_names = ["clip", "vae", "unet"]
    scheduler_func_names = [f"scheduler_step_{i}" for i in range(5)]

    # Build the main functions.
    if not os.path.exists(output):
        pkl_file = open("irmodule.pkl", "rb")
        relax_mod = pickle.load(pkl_file)
        pkl_file.close()

        relax_mod = relax.transform.RemoveUnusedFunctions(
            model_names + scheduler_func_names
        )(relax_mod)
        db = ms.database.create(work_dir=log_dir)
        with target, db, tvm.transform.PassContext(opt_level=3):
            relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)

        ex = relax.build(relax_mod, target)
        ex.mod.export_library(output)

    # Build the param transformation functions.
    if not os.path.exists(const_params_dir):
        pkl_file = open("irmodule.pkl", "rb")
        relax_mod = pickle.load(pkl_file)
        pkl_file.close()

        relax_mod = relax.transform.RemoveUnusedFunctions(
            [name + "_transform_params" for name in model_names]
        )(relax_mod)

        ex = relax.build(relax_mod, target="llvm")

        dev_cpu = tvm.cpu()
        vm_cpu = relax.vm.VirtualMachine(rt_mod=ex, device=dev_cpu)
        weight_dict = load_constant_from_file(
            "model_weights", dev=dev_cpu, deploy=False
        )
        param_dict = compute_params(vm_cpu, weight_dict)
        for model_name in model_names:
            dir_path = f"{const_params_dir}/{model_name}"
            os.makedirs(dir_path, exist_ok=True)
            for i, const_param in enumerate(param_dict[model_name]):
                pkl_const_param = open(f"{dir_path}/{model_name}_{i}.pkl", "wb")
                pickle.dump(const_param, file=pkl_const_param)
                pkl_const_param.close()


if __name__ == "__main__":
    ARGS = _parse_args()
    build(
        ARGS.target,
        ARGS.log_dir,
        ARGS.const_params_dir,
        ARGS.output,
    )
