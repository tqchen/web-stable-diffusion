

from diffusers import StableDiffusionPipeline
import torch
import torch._dynamo as dynamo
from diffusers.models.cross_attention import AttnProcessor2_0, CrossAttnProcessor

torch._dynamo.config.verbose=True



def export_models():
    class UNetWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, latent_model_input, timestep, encoder_hidden_states):
            return self.model(
                latent_model_input, timestep, encoder_hidden_states
            ).sample

    class CLIPWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids)[0]

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.safety_checker = None

    #clip_traced = dynamo.export(
    #    CLIPWrapper(pipe.text_encoder), torch.ones(1, 77, dtype=torch.int32)
    #)
    clip_traced = dynamo.export(
        CLIPWrapper(pipe.text_encoder), torch.ones(1, 77, dtype=torch.int32)
    )

    vae_dec_traced = dynamo.export(pipe.vae.decoder, torch.randn(1, 4, 64, 64))

    return
    # NOTE: Unet tracing is not yet working
    print("UNet")


    def processor(attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    pipe.unet.set_attn_processor(processor)

    unet_traced = dynamo.export(
        UNetWrapper(pipe.unet),
        *[torch.randn(2, 4, 64, 64), torch.tensor(1), torch.randn(2, 77, 768)],
    )

export_models()


