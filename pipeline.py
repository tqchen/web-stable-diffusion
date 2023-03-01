from typing import Optional

import torch
from transformers import CLIPTokenizer
from scheduler import PNDMScheduler
from tqdm import tqdm
from PIL import Image

import tvm
from tvm import relax


class TVMSDPipeline:
    def __init__(
        self,
        vm: relax.VirtualMachine,
        tokenizer: CLIPTokenizer,
        scheduler: PNDMScheduler,
        tvm_device,
        param_dict,
    ):
        def wrapper(f, params):
            def wrapped_f(*args):
                return f(*args, params)

            return wrapped_f

        self.vm = vm
        self.clip_to_text_embeddings = wrapper(vm["clip"], param_dict["clip"])
        self.unet_latents_to_noise_pred = wrapper(vm["unet"], param_dict["unet"])
        self.vae_to_image = wrapper(vm["vae"], param_dict["vae"])
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.tvm_device = tvm_device
        self.param_dict = param_dict

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        https://github.com/huggingface/diffusers/blob/2ab4fcdb43264ffd6e4c1824463425e621c967bf/src/diffusers/pipelines/pipeline_utils.py#L943
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

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
        text_embeddings = self.clip_to_text_embeddings(text_input_ids)[0]
        latents = torch.randn(
            # (batch_size * number_images_per_prompt, unet.in_channel, height // 8, width // 8)
            (1, 4, 64, 64),
            device="cpu",
            dtype=torch.float32,
        )
        latents = tvm.nd.array(latents.numpy(), self.tvm_device)

        for i in tqdm(range(num_inference_steps)):
            t = self.scheduler.scheduler_consts[i][0]
            noise_pred = self.unet_latents_to_noise_pred(latents, t, text_embeddings)
            latents = self.scheduler.step(self.vm, noise_pred, latents, i)

        image = self.vae_to_image(latents)
        image = image.numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return image
