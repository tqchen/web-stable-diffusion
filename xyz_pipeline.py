from typing import List, Optional, Union
import logging

import torch
from tqdm import tqdm
import pickle
from utils import numpy_to_pil

import tvm
from tvm import relax

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TVMSDPipeline:
    def __init__(
        self,
        wrapped,
        tokenizer,
        tvm_device,
    ):
        self._wrapped = wrapped
        self.clip_to_text_embeddings = wrapped.clip
        self.vae_to_image = wrapped.vae
        self.unet_latents_to_noise_pred = wrapped.unet

        self.tokenizer = tokenizer
        self.tvm_device = tvm_device
        self.scheduler = wrapped.scheduler

    def __call__(
        self,
        prompt: str,
        output_type: Optional[str] = "pil",
    ):
        # batch_size = 1
        # height = 512
        # width = 512
        num_inference_steps = 50

        # Using torch Tensor
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,  # 77
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(torch.int32)
        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        # Using TVM NDArray
        text_input_ids = tvm.nd.array(text_input_ids.cpu().numpy(), self.tvm_device)
        torch.save(torch.Tensor(text_input_ids.numpy()), "intermediate/clip_input.pt")

        text_embeddings = self.clip_to_text_embeddings(text_input_ids)
        latents = torch.randn(
            # (batch_size * number_images_per_prompt, unet.in_channel, height // 8, width // 8)
            (1, 4, 64, 64),
            device="cpu",
            dtype=torch.float32,
        )
        torch.save(torch.Tensor(text_embeddings.numpy()), "intermediate/clip_output.pt")

        latents = tvm.nd.array(latents.numpy(), self.tvm_device)

        for i in tqdm(range(num_inference_steps)):
            t = self.scheduler.scheduler_consts[i][0]
            torch.save(torch.Tensor(latents.numpy()), f"intermediate/unet_input_{i}.pt")
            noise_pred = self.unet_latents_to_noise_pred(latents, t, text_embeddings)
            torch.save(torch.Tensor(noise_pred.numpy()), f"intermediate/unet_output_{i}.pt")
            latents = self.scheduler.step(noise_pred, latents, i)
            torch.save(torch.Tensor(latents.numpy()), f"intermediate/scheduler_output_{i}.pt")

        torch.save(torch.Tensor(latents.numpy()), "intermediate/latents.pt")
        image = self.vae_to_image(latents)
        image = image.numpy()

        if output_type == "pil":
            image = numpy_to_pil(image)

        return image
