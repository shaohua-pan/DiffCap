# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import time

import numpy as np
import torch
import torch as th
from copy import deepcopy

import config
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
import torch.nn.functional as F
from model.smplify.run import smplify_runner
import articulate as art
from utils.kp_utils import *

# from model.smplify.temporal_smplify import TemporalSMPLify
# smplify = TemporalSMPLify(batch_size=60, step_size1=0.001, use_lbfgs=True)
p1, p2 = [], []
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        betas_slide=None,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2

        if betas_slide is not None and len(betas_slide) != 0:
            betas_slide = np.array(betas_slide, dtype=np.float64)
            self.betas_slide = betas_slide
            assert len(betas_slide.shape) == 2, "betas_slide must be 2-D"
            assert (betas_slide > 0).all() and (betas_slide <= 1).all()

            self.num_timesteps_slide = int(betas_slide.shape[0])

            alphas_slide = 1.0 - betas_slide
            self.alphas_cumprod_slide = np.cumprod(alphas_slide, axis=0)
            self.alphas_cumprod_prev_slide = np.concatenate((np.ones((1, len(self.alphas_cumprod_slide[0]))), self.alphas_cumprod_slide[:-1, :]), axis=0)
            self.alphas_cumprod_next_slide = np.concatenate((self.alphas_cumprod_slide[1:, :], np.zeros((1, len(self.alphas_cumprod_slide[0])))), axis=0)
            assert self.alphas_cumprod_prev_slide.shape[0] == self.num_timesteps_slide

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_alphas_cumprod_slide = np.sqrt(self.alphas_cumprod_slide)
            self.sqrt_one_minus_alphas_cumprod_slide = np.sqrt(1.0 - self.alphas_cumprod_slide)
            self.log_one_minus_alphas_cumprod_slide = np.log(1.0 - self.alphas_cumprod_slide)
            self.sqrt_recip_alphas_cumprod_slide = np.sqrt(1.0 / self.alphas_cumprod_slide)
            self.sqrt_recipm1_alphas_cumprod_slide = np.sqrt(1.0 / self.alphas_cumprod_slide - 1)

            # calculations for posterior q(x_{t-1} | x_t, x_0)
            # self.posterior_variance_slide = (
            #         betas_slide * (1.0 - self.alphas_cumprod_prev_slide) / (1.0 - self.alphas_cumprod_slide)
            # )
            # # log calculation clipped because the posterior variance is 0 at the
            # # beginning of the diffusion chain.
            # self.posterior_log_variance_clipped_slide = np.log(
            #     np.append(self.posterior_variance_slide[1], self.posterior_variance_slide[1:])
            # )
            # self.posterior_mean_coef1 = (
            #         betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            # )
            # self.posterior_mean_coef2 = (
            #         (1.0 - self.alphas_cumprod_prev)
            #         * np.sqrt(alphas)
            #         / (1.0 - self.alphas_cumprod)
            # )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_sample_sliding(self, x_start, t, noise=None, diff_t=15):
        """
        Diffuse the dataset for a given number of diffusion steps, diffuse different steps for the different frame of a sequence.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param diff_t: the number of diffusion steps difference to be diffused for each frame of a sequence.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        x_start = x_start.permute(0, 3, 1, 2)
        noise = noise.permute(0, 3, 1, 2)
        assert noise.shape == x_start.shape
        res = (
            _extract_into_tensor_sliding(self.sqrt_alphas_cumprod, t, x_start.shape, diff_t) * x_start
            + _extract_into_tensor_sliding(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, diff_t)
            * noise
        )
        return res.permute(0, 2, 3, 1)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # Fix inpainting mask
        if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
            inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
            ti = t[0].item()
            if ti == 0 and x.shape[2] == 6:
                pass
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
            model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)

        # Adaptive inpainting, calculated by the kp's confidence and time step
        elif 'inpainted_motion' in model_kwargs['y'].keys() and 'inpainting_conf' in model_kwargs['y'].keys():
            inpainting_conf, inpainted_motion = model_kwargs['y']['inpainting_conf'], model_kwargs['y']['inpainted_motion']
            ti = t[0].item()
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert model_output.shape == inpainted_motion.shape == inpainting_conf.shape
            inpainting_conf[inpainting_conf < 0.999] = 0
            ct = ti / 4.0
            inpainting_conf = inpainting_conf * ct
            # inpainting_conf = inpainting_conf.permute(0, 3, 1, 2)
            # inpainting_conf[torch.mean(inpainting_conf, dim=[2, 3]) < 0.7] = 0
            # inpainting_conf = inpainting_conf.permute(0, 2, 3, 1)
            model_output = (model_output * (1 - inpainting_conf)) + (inpainted_motion * inpainting_conf)

        # Use IMU to inpaint the pose, x = (poser 24*6, j2dc 33*2, accr 6*3)
        if 'accr' in model_kwargs['y'].keys() and 'orir' in model_kwargs['y'].keys():
            inpainted_imu_acc = model_kwargs['y']['accr']
            inpainted_imu_ori = model_kwargs['y']['orir']
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            model_output = model_output.permute(0, 3, 1, 2)  # [bs, njoints, nfeats, nframes] -> [bs, nframes, njoints, nfeats]
            model_output[:, :, :, -6*3:] = inpainted_imu_acc
            model_output_poser = model_output[:, :, :, :24*6].reshape(model_output.shape[0], model_output.shape[1], 24, 6)
            model_output_poser[:, :, config.ji_mask, :] = inpainted_imu_ori.reshape(model_output.shape[0], model_output.shape[1], 6, 6)
            model_output[:, :, :, :24 * 6] = model_output_poser.reshape(model_output.shape[0], model_output.shape[1], 1, 24 * 6)
            model_output = model_output.permute(0, 2, 3, 1)  # [bs, nframes, njoints, nfeats] -> [bs, njoints, nfeats, nframes]

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_dual_mean_variance(
        self, model1, model2, x1, x2, t, clip_denoised=True, denoised_fn=None, model_kwargs1=None, model_kwargs2=None
    ):
        ti = t[0].item()
        if model_kwargs1 is None:
            model_kwargs1 = {}
        if model_kwargs2 is None:
            model_kwargs2 = {}
        B, C = x1.shape[:2]
        assert t.shape == (B,)
        model_output1 = model1(x1, self._scale_timesteps(t), **model_kwargs1)
        model_output2 = model2(x2, self._scale_timesteps(t), **model_kwargs2)
        if 'position' in model_kwargs2['y'].keys():
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred'
            inpainted_motion, oric, K, tstart = model_kwargs2['y']['inpainted_motion'], model_kwargs2['y']['oric'], model_kwargs2['y']['cam_k'], model_kwargs2['y']['tstart']
            position, contact = model_kwargs2['y']['position'], model_kwargs2['y']['contact']
            floor = model_kwargs2['y']['floor']
            Rwc = model_kwargs2['y']['Rcw'].T
            j2dc_opt = inpainted_motion.reshape(-1, 33, 3)
            oric = oric.reshape(-1, 6, 3, 3)
            pose, tran = model_output1.permute(0, 3, 1, 2), model_output2.permute(0, 3, 1, 2)
            posec = art.math.r6d_to_rotation_matrix(pose.reshape(-1, 24, 6)).reshape(-1, 24, 3, 3)
            posec = config.body_model.inverse_kinematics_R(posec).reshape(-1, 24, 3, 3)
            posec[:, 0] = oric[:, -1]
            tran = torch.matmul(oric[:, -1], tran.reshape(-1, 3, 1)).reshape(-1, 3) * config.vel_scale / 60
            tran = tran.reshape(-1, 3)

            # use contact to update the translation
            pfoot = config.body_model.forward_kinematics(posec)[1][:, 10:12] # (n_frame, 2, 3)
            pfoot_v = torch.zeros_like(pfoot)
            pfoot_v[1:] = pfoot[:-1] - pfoot[1:]  # (n_frame, 2, 3)
            pfoot_v = art.math.lerp(pfoot_v[:, 0], pfoot_v[:, 1], contact.max(dim=1).indices.view(-1, 1))
            weight = (contact.max(dim=1).values.clamp(0.5, 0.9) - 0.5) / 0.4
            tran = tran * (1 - weight.unsqueeze(-1)) + pfoot_v * weight.unsqueeze(-1)

            # complementary filter of the translation
            position = position.reshape(-1, 3)
            c = j2dc_opt[..., -1].mean(dim=1)
            tran_lerp = torch.zeros(position.shape).to(position.device)
            for i in range(len(position)):
                if i == 0:
                    tranv = tran[0] + tstart
                else:
                    tranv = tran_lerp[i - 1] + tran[i]
                k = (c[i] - config.conf_range[0]) / (config.conf_range[1] - config.conf_range[0])
                if k > 1: k = 1
                if (position[i] - tranv).norm() > config.distrance_threshold or config.tran_filter_num >= 1:
                    tran_lerpi = position[i]
                elif c[i] >= config.conf_range[1]:
                    tran_lerpi = art.math.lerp(tranv, position[i], config.tran_filter_num * k)
                else:
                    tran_lerpi = tranv

                # add gravity
                if floor is not None:
                    tranw = torch.matmul(Rwc, tran_lerpi.reshape(3, 1)).reshape(3)
                    tranw[1] += config.gravity
                    pfootw = torch.matmul(Rwc, pfoot.unsqueeze(-1)).squeeze(-1)[i] + tranw.reshape(1, 3)
                    down_floor = floor - pfootw[1]
                    tranw[1] += torch.max(down_floor[1], torch.zeros_like(down_floor[1]))
                    tran_lerpi = torch.matmul(Rwc.T, tranw.reshape(3, 1)).reshape(3)
                tran_lerp[i] = tran_lerpi

            tran = tran_lerp

            # optimize the pose and translation
            # if ti <= 1 and j2dc_opt[..., -1].mean() > config.conf_range[0]:
            #     posec, tran, _ = smplify_runner(posec, tran, j2dc_opt, oric, batch_size=posec.shape[0], lr=0.0001, use_lbfgs=True, opt_steps=1, cam_k=K, use_head=True)

            posec[:, 0] = torch.eye(3)
            pose = config.body_model.forward_kinematics_R(posec).reshape(-1, 24, 3, 3)
            pose = art.math.rotation_matrix_to_r6d(pose).reshape(1, -1, 24, 6)
            tran[1:] = tran[1:] - tran[:-1]
            tran[0] -= tstart
            tran = torch.matmul(oric[:, -1].transpose(1, 2), tran.reshape(-1, 3, 1)).reshape(-1, 3) / config.vel_scale * 60
            model_output1, model_output2 = pose.permute(0, 2, 3, 1), tran.reshape(1, -1, 1, 3).permute(0, 2, 3, 1)
        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]

        model_variance1 = _extract_into_tensor(model_variance, t, x1.shape)
        model_log_variance1 = _extract_into_tensor(model_log_variance, t, x1.shape)
        model_variance2 = _extract_into_tensor(model_variance, t, x2.shape)
        model_log_variance2 = _extract_into_tensor(model_log_variance, t, x2.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        pred_xstart1 = process_xstart(model_output1)
        pred_xstart2 = process_xstart(model_output2)
        model_mean1, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart1, x_t=x1, t=t
        )
        model_mean2, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart2, x_t=x2, t=t
        )
        assert (
            model_mean1.shape == model_log_variance1.shape == pred_xstart1.shape == x1.shape
        ) and (
            model_mean2.shape == model_log_variance2.shape == pred_xstart2.shape == x2.shape
        )
        return {
            "mean": model_mean1,
            "variance": model_variance1,
            "log_variance": model_log_variance1,
            "pred_xstart": pred_xstart1,
        }, {
            "mean": model_mean2,
            "variance": model_variance2,
            "log_variance": model_log_variance2,
            "pred_xstart": pred_xstart2,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_xstart_sliding(self, x_t, t, pred_xstart):
        x_t = x_t.permute(0, 3, 1, 2)
        pred_xstart = pred_xstart.permute(0, 3, 1, 2)
        res =  (
            _extract_into_tensor_sliding_test(self.sqrt_recip_alphas_cumprod_slide, t, x_t.shape, diff_t=15) * x_t
            - pred_xstart
        ) / _extract_into_tensor_sliding(self.sqrt_recipm1_alphas_cumprod_slide, t, x_t.shape, diff_t=15)
        return res.permute(0, 2, 3, 1)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param const_noise: if True, use the same noise for all samples.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param skip_timesteps: if specified, skip this number of timesteps.
        :param init_image: if specified, use this as the initial image, this image will be noised to timestep T.
        :param randomize_class: if True, will randomize the class of the image at each timestep (not used).
        :param cond_fn_with_grad: if True, will use the gradient of the cond_fn to sample (not used).
        :param dump_steps: if specified, will dump the samples at these timesteps.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling.
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
        )):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_dual_sample(
        self,
        model1,
        model2,
        x1,
        x2,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs1=None,
        model_kwargs2=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_dual_mean_variance(
            model1,
            model2,
            x1,
            x2,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs1=model_kwargs1,
            model_kwargs2=model_kwargs2,
        )
        out1, out2 = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps1 = self._predict_eps_from_xstart(x1, t, out1["pred_xstart"])
        eps2 = self._predict_eps_from_xstart(x2, t, out2["pred_xstart"])

        alpha_bar1 = _extract_into_tensor(self.alphas_cumprod, t, x1.shape)
        alpha_bar_prev1 = _extract_into_tensor(self.alphas_cumprod_prev, t, x1.shape)
        alpha_bar2 = _extract_into_tensor(self.alphas_cumprod, t, x2.shape)
        alpha_bar_prev2 = _extract_into_tensor(self.alphas_cumprod_prev, t, x2.shape)
        sigma1 = (
            eta
            * th.sqrt((1 - alpha_bar_prev1) / (1 - alpha_bar1))
            * th.sqrt(1 - alpha_bar1 / alpha_bar_prev1)
        )
        sigma2 = (
            eta
            * th.sqrt((1 - alpha_bar_prev2) / (1 - alpha_bar2))
            * th.sqrt(1 - alpha_bar2 / alpha_bar_prev2)
        )
        # Equation 12.
        noise1 = th.randn_like(x1)
        noise2 = th.randn_like(x2)
        mean_pred1 = (
            out1["pred_xstart"] * th.sqrt(alpha_bar_prev1)
            + th.sqrt(1 - alpha_bar_prev1 - sigma1 ** 2) * eps1
        )
        mean_pred2 = (
            out2["pred_xstart"] * th.sqrt(alpha_bar_prev2)
            + th.sqrt(1 - alpha_bar_prev2 - sigma2 ** 2) * eps2
        )
        nonzero_mask1 = (
            (t != 0).float().view(-1, *([1] * (len(x1.shape) - 1)))
        )  # no noise when t == 0
        nonzero_mask2 = (
            (t != 0).float().view(-1, *([1] * (len(x2.shape) - 1)))
        )  # no noise when t == 0
        sample1 = mean_pred1 + nonzero_mask1 * sigma1 * noise1
        sample2 = mean_pred2 + nonzero_mask2 * sigma2 * noise2
        return {"sample": sample1, "pred_xstart": out1["pred_xstart"]}, {"sample": sample2, "pred_xstart": out2["pred_xstart"]}

    def ddim_sample_slide(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        # base_timestep = model_kwargs['y']['base_timesteps']
        # timestep = torch.tensor(base_timestep, device=t.device)[t]
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        if 'y' in model_kwargs and 'slide_first' in model_kwargs['y']:
            indices = torch.tensor([0, 3, 6, 9, 999]).to(t.device)
        else:
            indices = torch.tensor([0, 3, 6, 9, 12]).to(t.device)
        t_eps = (t // 3)
        if t_eps == 5 and 'y' in model_kwargs and 'slide_first' in model_kwargs['y']:
            eps = self._predict_eps_from_xstart(x, indices[t_eps], out["pred_xstart"])
            model_kwargs['y'].pop('slide_first')
        else:
            eps = self._predict_eps_from_xstart_sliding(x, indices[t_eps], out["pred_xstart"])
        x = x.permute(0, 3, 1, 2)
        alpha_bar = _extract_into_tensor_sliding(self.alphas_cumprod, t, x.shape, diff_t=15).permute(0, 2, 3, 1)
        alpha_bar_prev = _extract_into_tensor_sliding(self.alphas_cumprod_prev, t, x.shape, diff_t=15).permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 1)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_mul(
        self,
        model,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        q=None,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        ):
            final = sample
        q.put(final["sample"])

    def ddim_dual_sample_loop(
        self,
        model1,
        model2,
        shape1,
        shape2,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs1=None,
        model_kwargs2=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_dual_sample_loop_progressive(
            model1,
            model2,
            shape1,
            shape2,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs1=model_kwargs1,
            model_kwargs2=model_kwargs2,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
        ):
            final = sample
        return final[0]["sample"], final[1]["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        if 'y' in model_kwargs.keys() and 'slide' in model_kwargs['y'].keys():
            indices = [0, 3, 6, 9, 12][::-1]

        if init_image is not None:
            # my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            # img = self.q_sample(init_image, my_t, img)
            img = init_image

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                sample_fn = self.ddim_sample_slide if 'y' in model_kwargs.keys() and 'slide' in model_kwargs['y'] else sample_fn
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def ddim_dual_sample_loop_progressive(
        self,
        model1,
        model2,
        shape1,
        shape2,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs1=None,
        model_kwargs2=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model1.parameters()).device
        assert isinstance(shape1, (tuple, list)) and isinstance(shape2, (tuple, list))
        if noise is not None:
            img1, img2 = noise
        else:
            img1, img2 = th.randn(*shape1, device=device), th.randn(*shape2, device=device)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            img1, img2 = init_image

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape1[0], device=device)
            with th.no_grad():
                sample_fn = self.ddim_dual_sample
                out = sample_fn(
                    model1,
                    model2,
                    img1,
                    img2,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs1=model_kwargs1,
                    model_kwargs2=model_kwargs2,
                    eta=eta,
                )
                yield out
                img1, img2 = out[0]["sample"], out[1]["sample"]

    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        """
        Sample x_{t-1} from the model using Pseudo Linear Multistep.

        Same usage as p_sample().
        """
        if not int(order) or not 1 <= order <= 4:
            raise ValueError('order is invalid (should be int from 1-4).')

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                else:
                    out = out_orig

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            # Pseudo Improved Euler
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime
        else:
            # Pseudo Linear Multistep (Adams-Bashforth)
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] + 5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] + 37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError('cur_order is invalid.')
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "old_eps": old_eps}

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Generate samples from the model using Pseudo Linear Multistep.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from each
        timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses_dual(self, model, x_start, t, model_kwargs=None, noise=None, noise_dot=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        x_start_dot = x_start[1]
        x_start = x_start[0]
        if noise is None:
            noise = th.randn_like(x_start)
            noise_dot = th.randn_like(x_start_dot)
        x_t = self.q_sample(x_start, t, noise=noise)
        x_t_dot = self.q_sample(x_start_dot, t, noise=noise_dot)
        terms = {}
        model_output, model_output_dot = model((x_t, x_t_dot), self._scale_timesteps(t), **model_kwargs)
        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        target_dot = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start_dot, x_t=x_t_dot, t=t
            )[0],
            ModelMeanType.START_X: x_start_dot,
            ModelMeanType.EPSILON: noise_dot,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
        assert model_output_dot.shape == target_dot.shape == x_start_dot.shape  # [bs, njoints, nfeats, nframes]
        terms["loss"] = 1000 * mean_flat(self.l2_loss(target, model_output)) + 1000 * mean_flat(self.l2_loss(target_dot, model_output_dot))
        return terms

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if isinstance(x_start, tuple):
            return self.training_losses_dual(model, x_start, t, model_kwargs=None, noise=None)
        if noise is None:
            noise = th.randn_like(x_start)
        if 'sliding' in model_kwargs.keys():
            x_t = self.q_sample_sliding(x_start, t, noise=noise)
            model_kwargs.pop('sliding')
        else:
            x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        # only use MSE
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

            # giving a loss const of 1000 to the MSE term
            if 'y' in model_kwargs.keys() and 'vel' in model_kwargs['y'].keys():
                terms["mse"] = 1000 * mean_flat(self.l2_loss(target, model_output))
            else:
                terms["mse"] = 1000 * mean_flat(self.l2_loss(target, model_output))

            terms["loss"] = terms["mse"] + terms.get('vb', 0.)
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _extract_into_tensor_sliding(arr, timesteps, broadcast_shape, diff_t):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :param diff_t: the difference between different frames.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    diff = th.from_numpy(np.arange(broadcast_shape[1]) * diff_t).to(device=timesteps.device).long().unsqueeze(0)
    t = timesteps.unsqueeze(1) + diff
    res = th.from_numpy(arr).to(device=timesteps.device)[t].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def _extract_into_tensor_sliding_test(arr, timesteps, broadcast_shape, diff_t):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :param diff_t: the difference between different frames.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    diff = th.from_numpy(np.arange(broadcast_shape[1]) * diff_t).to(device=timesteps.device).long().unsqueeze(0)
    t = timesteps.unsqueeze(1) + diff
    res = th.from_numpy(arr).to(device=timesteps.device)[t].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

jo = config.body_model.get_zero_pose_joint_and_vertex()[0]
bo = config.body_model.joint_position_to_bone_vector(jo.unsqueeze(0)).view(24, 3, 1).to(config.device)
def forward_kinematics(p):
    p = art.math.r6d_to_rotation_matrix(p.contiguous()).view(-1, 24, 3, 3)
    pb = torch.stack([p[:, config.body_model.parent[i]].matmul(bo[i]) for i in range(1, 24)], dim=1)
    pb = torch.cat((torch.zeros(p.shape[0], 1, 3, device=config.device), pb.squeeze(-1)), dim=1)
    return config.body_model.bone_vector_to_joint_position(pb)