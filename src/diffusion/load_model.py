import json

import torch
import math

from ..vae.model_autoencoder import ConvAutoencoder, NoAutoencoder
from .gaussian_diffusion import GaussianDiffusion
from .model_dhariwal import UNetModel


def construct_model_from_json(
        model_json,
        default_chunk_size=64,
        default_channels=1):

    with open(model_json, "r") as json_file:
        structure = json.load(json_file)

    if "vae" not in structure:
        vae = NoAutoencoder()
        chunk_size = latent_size = default_chunk_size
        latent_channels = default_channels
        if "data_downsample_factor" in structure["model"]:
            factor = structure["model"]["data_downsample_factor"]

            # take log 2 to get the equivalent in n blocks, should be an integer
            floorplan_downsamples = int(math.log2(factor))

            latent_size = default_chunk_size // factor
        else:
            floorplan_downsamples = 0
    else:
        chunk_size = structure["vae"]["chunk_size"]
        latent_size = structure["vae"]["latent_size"]
        latent_channels = structure["vae"]["latent_channels"]
        vae_dim = structure["vae"]["dim"]
        floorplan_downsamples = n_blocks = structure["vae"]["n_blocks"]

        vae = ConvAutoencoder(latent_channels=latent_channels,
                              process_channels=vae_dim,
                              num_scaling_blocks=n_blocks)
        vae.load_state_dict(torch.load(structure["vae"]["path"]))
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()

    use_new_attention_order = structure["model"]["use_new_attention_order"] if "use_new_attention_order" in structure["model"] else False
    use_scale_shift_norm = structure["model"]["use_scale_shift_norm"] if "use_scale_shift_norm" in structure["model"] else False
    add_conditions_to_input = structure["model"]["add_conditions_to_input"] if "add_conditions_to_input" in structure["model"] else False
    add_flipped_conditions = structure["model"]["add_flipped_conditions"] if "add_flipped_conditions" in structure["model"] else [
    ]
    floorplan_input = structure["model"]["floorplan_input"] if "floorplan_input" in structure["model"] else False
    boundary_input = structure["model"]["boundary_input"] if "boundary_input" in structure["model"] else False
    height_conditioning = structure["model"]["height_conditioning"] if "height_conditioning" in structure["model"] else True

    super_scaling = structure["model"]["is_super_scaling"] if "is_super_scaling" in structure["model"] else False

    arch = UNetModel(
        latent_size,
        latent_channels,
        structure["model"]["dim"],
        latent_channels,
        structure["model"]["num_res_blocks"],
        structure["model"]["self_att_resolutions"],
        num_heads=structure["model"]["heads"],
        channel_mult=structure["model"]["dim_mults"],
        dims=3,
        cross_attention_resolutions=structure["model"]["cross_att_resolutions"],
        num_conditions=structure["model"]["num_conditions"],
        condition_channels=latent_channels,
        dropout=structure["model"]["dropout_prob"],
        use_new_attention_order=use_new_attention_order,
        use_scale_shift_norm=use_scale_shift_norm,
        add_conditions_to_input=add_conditions_to_input,
        add_flipped_conditions=add_flipped_conditions,
        use_floorplan_conditions=floorplan_input,
        floorplan_downsamples=floorplan_downsamples,
        use_boundary_conditions=boundary_input,
        use_height_conditioning=height_conditioning,
        is_super_scaling=super_scaling
    )

    sigma_min = structure["model"]["sigma_min"]
    sigma_max = structure["model"]["sigma_max"]
    cfg = structure["model"]["cfg"] if "cfg" in structure["model"] else False
    cfg_mode = structure["model"]["cfg_mode"] if "cfg_mode" in structure["model"] else "single"

    model = GaussianDiffusion(
        model=arch,
        vae=vae,
        cfg=cfg,
        cfg_mode=cfg_mode,
        preconditioning=structure["model"]["preconditioning"],
        sigma_min=sigma_min,
        sigma_max=sigma_max
    )

    # chunk size is in the size of the data
    return arch, model, chunk_size
