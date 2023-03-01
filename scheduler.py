from typing import List

import numpy as np
import json

import tvm
from tvm import relax


class PNDMScheduler:
    def __init__(self, dev):
        with open("scheduler_consts.json", "r") as file:
            jsoncontent = file.read()
            _scheduler_consts = json.loads(jsoncontent)
            self.scheduler_consts: List[List[tvm.nd.NDArray]] = []
            for t in range(len(_scheduler_consts)):
                timestep = tvm.nd.array(
                    np.array(_scheduler_consts[t][0], dtype="int32"), device=dev
                )
                consts = [timestep]
                for i in range(1, 4):
                    consts.append(
                        tvm.nd.array(
                            np.array(_scheduler_consts[t][i], dtype="float32"),
                            device=dev,
                        )
                    )
                self.scheduler_consts.append(consts)

        self.ets: List[tvm.nd.NDArray] = []
        self.cur_sample: tvm.nd.NDArray

    def step(
        self,
        vm: relax.VirtualMachine,
        model_output: tvm.nd.NDArray,
        sample: tvm.nd.NDArray,
        counter: int,
    ) -> tvm.nd.NDArray:
        if counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)

        consts = self.scheduler_consts[counter]
        sample_coeff = consts[1]
        alpha_diff = consts[2]
        model_output_denom_coeff = consts[3]

        if counter == 0:
            self.cur_sample = sample
            prev_latents = vm["scheduler_step_0"](
                model_output,
                sample,
                sample_coeff,
                alpha_diff,
                model_output_denom_coeff,
            )
        elif counter == 1:
            prev_latents = vm["scheduler_step_1"](
                model_output,
                self.cur_sample,
                sample_coeff,
                alpha_diff,
                model_output_denom_coeff,
                self.ets[-1],
            )
        elif counter == 2:
            prev_latents = vm["scheduler_step_2"](
                sample,
                sample_coeff,
                alpha_diff,
                model_output_denom_coeff,
                self.ets[-2],
                self.ets[-1],
            )
        elif counter == 3:
            prev_latents = vm["scheduler_step_3"](
                sample,
                sample_coeff,
                alpha_diff,
                model_output_denom_coeff,
                self.ets[-3],
                self.ets[-2],
                self.ets[-1],
            )
        else:
            prev_latents = vm["scheduler_step_4"](
                sample,
                sample_coeff,
                alpha_diff,
                model_output_denom_coeff,
                self.ets[-4],
                self.ets[-3],
                self.ets[-2],
                self.ets[-1],
            )

        return prev_latents[0]
