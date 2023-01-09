import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(
                n_timestep + 1,
                dtype=torch.float64,
            )
            / n_timestep
            + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_maxx=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown")
    return betas.numpy()


def make_ddim_timesteps(
    ddim_discr_method, num_ddim_timestep, num_ddpm_timesteps, verbose=True
):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timestep
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timestep)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f"there is no  ddim discretization method called {ddim_discr_method}"
        )
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"selected timestep for ddim sampler: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the formula provided https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    if verbose:
        print(f"selected alphasa for ddim sampler at: {alphas}; a_(t-1): {alphas_prev}")
        print(
            f"for the chosen value of eta, which is {eta}, "
            f"this result in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    create a beta  schedule that discrete the given alpha_t bar functionm
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    parameter:
        num_diffusion_timesteps: the number of betas to produce
        alpha_bar: a lambda that takes an argument t from 0 to 1 and
                    produce  the cumulative product of (1-bet) up to that
                    part of the diffusion process
        max_beta: the maximum bar to use; use values lower than 1 to
                    prevent singularities
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,)) * (len(x_shape) - 1))


def checkpoint(func, inputs, params, flag):
    """
    check evaluate without caching the intermediate activations, allowing for
    reduced memory at expensse of extra compute in the backward pass.
    parameter:
        func: the function to evaluate
        inputs: the argument sequence to pass the 'func'
        paraams: a sequence of parameters 'func' depends on but does not
                explicitly  take as arguments
        flag: if false, disable gradient checkpointing
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.FUnction):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_funcion = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    create timestep embeddings
    parameter:
        timestep: 1-D Tensor of N indices, one per batch element.
                these may be fractional
        dim: the dimension of the output
        max_period: controls the minimum frequency of the embeddings
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    make standard normalization layer
    """
    return GroupNorm32(32, channels)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimension: {dims}")


class HybridConditioner(nn.Module):
    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {"c_concat": [c_concat], "c_crossattn": [c_crossattn]}


def noise_like(x, repeat=False):
    noise = torch.randn_like(x)
    if repeat:
        bs = x.shape[0]
        noise = noise[0:1].repeat(bs, *((1,) * (len(x.shape) - 1)))
    return noise


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params
