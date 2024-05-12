
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, xray, ct):

        t = torch.randint(self.T, size=(xray.shape[0], ), device=xray.device)
        noise_xray = torch.randn_like(xray)
        noise_ct = torch.randn_like(ct)
        xray_t = (
                extract(self.sqrt_alphas_bar, t, xray.shape) * xray +
                extract(self.sqrt_one_minus_alphas_bar, t, xray.shape) * noise_xray)
        ct_t = (
                extract(self.sqrt_alphas_bar, t, ct.shape) * ct +
                extract(self.sqrt_one_minus_alphas_bar, t, ct.shape) * noise_ct)

        result_xray, result_ct = self.model(xray_t, ct_t, t)

        lossct = F.mse_loss(result_ct, noise_ct, reduction='none').sum()

        lossxray = F.mse_loss(result_xray, noise_xray, reduction='none').sum()

        return lossxray + lossct


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, xray_t, t_xray, ct_t):
        var_ct = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var_ct = extract(var_ct, t_xray, ct_t.shape)
        var_xray = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var_xray = extract(var_xray, t_xray, xray_t.shape)

        eps_xray, eps_ct = self.model(xray_t, ct_t, t_xray)

        ct_t_prev_mean = self.predict_xt_prev_mean_from_eps(ct_t, t_xray, eps=eps_ct)
        xray_t_prev_mean = self.predict_xt_prev_mean_from_eps(xray_t, t_xray, eps=eps_xray)

        return xray_t_prev_mean, var_xray, ct_t_prev_mean, var_ct

    def forward(self, xray_T, ct_T):

        xray_t = xray_T
        ct_t = ct_T

        xray_fixed = xray_T

        for time_step in reversed(range(self.T)):
            # print(time_step)

            noise_xray = torch.randn_like(xray_fixed)

            t_xray = xray_t.new_ones([xray_T.shape[0], ], dtype=torch.long) * time_step

            xray_fixed = (
                    extract(self.sqrt_alphas_bar, t_xray, xray_fixed.shape) * xray_fixed +
                    extract(self.sqrt_one_minus_alphas_bar, t_xray, xray_fixed.shape) * noise_xray)

            xray_mean, xray_var, ct_mean, ct_var = self.p_mean_variance(xray_t=xray_fixed, t_xray=t_xray, ct_t=ct_t)

            if time_step > 0:
                noise_ct = torch.randn_like(ct_t)
                noise_xray = torch.randn_like(xray_t)
            else:
                noise_ct = 0
                noise_xray = 0
            ct_t = ct_mean + torch.sqrt(ct_var) * noise_ct

            assert torch.isnan(ct_t).int().sum() == 0, "nan in tensor."
            assert torch.isnan(xray_t).int().sum() == 0, "nan in tensor."
            xray_fixed = xray_T
        ct_0 = ct_t
        xray_0 = xray_fixed
        return torch.clip(xray_0, -1, 1), torch.clip(ct_0, -1, 1)


class GaussianDiffusionSampler_reverse(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, xray_t, t_xray, ct_t):
        var_ct = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var_ct = extract(var_ct, t_xray, ct_t.shape)
        var_xray = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var_xray = extract(var_xray, t_xray, xray_t.shape)

        eps_xray, eps_ct = self.model(xray_t, ct_t, t_xray)

        ct_t_prev_mean = self.predict_xt_prev_mean_from_eps(ct_t, t_xray, eps=eps_ct)
        xray_t_prev_mean = self.predict_xt_prev_mean_from_eps(xray_t, t_xray, eps=eps_xray)

        return xray_t_prev_mean, var_xray, ct_t_prev_mean, var_ct

    def forward(self, xray_T, ct_T):

        xray_t = xray_T
        ct_t = ct_T
        xray_fixed = xray_T

        xray_t_temp = xray_T
        ct_t_temp = ct_T
        xray_fixed_temp = xray_T
        
        for time_step in reversed(range(self.T)):
            print(time_step)
            noise_xray = torch.randn_like(xray_fixed)

            t_xray = xray_t.new_ones([xray_T.shape[0], ], dtype=torch.long) * time_step
            xray_fixed = (
                    extract(self.sqrt_alphas_bar, t_xray, xray_fixed.shape) * xray_fixed + 
                    extract(self.sqrt_one_minus_alphas_bar, t_xray, xray_fixed.shape) * noise_xray)
            xray_mean, xray_var, ct_mean, ct_var = self.p_mean_variance(xray_t=xray_fixed, t_xray=t_xray, ct_t=ct_t)

            # no noise when t == 0
            if time_step > 0:
                noise_ct = torch.randn_like(ct_t)
            else:
                noise_ct = 0
            ct_t = ct_mean + torch.sqrt(ct_var) * noise_ct

            assert torch.isnan(ct_t).int().sum() == 0, "nan in tensor."
            assert torch.isnan(xray_t).int().sum() == 0, "nan in tensor."

            xray_fixed = xray_T
            if (time_step + 1) % 40 == 0 and time_step > 0 and (time_step + 1) != 1000:
                for time_step_temp in reversed(range(40)):
                    print("Reverse!")
                    print(time_step_temp)
                    t_xray_temp = xray_t.new_ones([xray_T.shape[0], ], dtype=torch.long) * time_step_temp

                    noise_xray_temp = torch.randn_like(xray_fixed)
                    xray_fixed_temp = (
                            extract(self.sqrt_alphas_bar, t_xray_temp + t_xray, xray_fixed_temp.shape) * xray_fixed_temp
                            extract(self.sqrt_one_minus_alphas_bar, t_xray_temp + t_xray, xray_fixed_temp.shape) * noise_xray_temp)

                    noise_ct_temp = torch.randn_like(ct_t)
                    ct_t = (
                            extract(self.sqrt_alphas_bar, t_xray_temp,
                                    ct_t.shape) * ct_t +
                            extract(self.sqrt_one_minus_alphas_bar, t_xray_temp,
                                    ct_t.shape) * noise_ct_temp)
                    xray_mean_temp, xray_var_temp, ct_mean_temp, ct_var_temp = self.p_mean_variance(xray_t=xray_fixed_temp, t_xray=t_xray_temp + t_xray,
                                                                                ct_t=ct_t)

                    # no noise when t == 0
                    if time_step > 0:
                        noise_ct_temp = torch.randn_like(ct_t)
                    else:
                        noise_ct_temp = 0
                    ct_t = ct_mean_temp + torch.sqrt(ct_var_temp) * noise_ct_temp

                    assert torch.isnan(ct_t).int().sum() == 0, "nan in tensor."
                    assert torch.isnan(xray_t).int().sum() == 0, "nan in tensor."

                    xray_fixed_temp = xray_T

        ct_0 = ct_t
        xray_0 = xray_fixed
        return torch.clip(xray_0, -1, 1), torch.clip(ct_0, -1, 1)

class GaussianDiffusionSampler1(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, xray_t, t_xray, ct_t):

        var_ct = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var_ct = extract(var_ct, t_xray, ct_t.shape)
        var_xray = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var_xray = extract(var_xray, t_xray, xray_t.shape)

        eps_xray, eps_ct = self.model(xray_t, ct_t, t_xray)

        ct_t_prev_mean = self.predict_xt_prev_mean_from_eps(ct_t, t_xray, eps=eps_ct)
        xray_t_prev_mean = self.predict_xt_prev_mean_from_eps(xray_t, t_xray, eps=eps_xray)

        return xray_t_prev_mean, var_xray, ct_t_prev_mean, var_ct

    def forward(self, xray_T, ct_T):

        xray_t = xray_T
        ct_t = ct_T

        ct_fixed = ct_T

        for time_step in reversed(range(self.T)):
            print(time_step)

            noise_ct = torch.randn_like(ct_fixed)

            t_ct = ct_t.new_ones([ct_T.shape[0], ], dtype=torch.long) * time_step

            ct_fixed = (
                    extract(self.sqrt_alphas_bar, t_ct, ct_fixed.shape) * ct_fixed + 
                    extract(self.sqrt_one_minus_alphas_bar, t_ct, ct_fixed.shape) * noise_ct)

            xray_mean, xray_var, ct_mean, ct_var = self.p_mean_variance(xray_t=xray_t, t_xray=t_ct, ct_t=ct_fixed)

            if time_step > 0:
                noise_ct = torch.randn_like(ct_t)
                noise_xray = torch.randn_like(xray_t)
            else:
                noise_ct = 0
                noise_xray = 0
            xray_t = xray_mean + torch.sqrt(xray_var) * noise_xray
            # xray_t = xray_mean + torch.sqrt(xray_var) * noise_xray

            assert torch.isnan(ct_t).int().sum() == 0, "nan in tensor."
            assert torch.isnan(xray_t).int().sum() == 0, "nan in tensor."
            ct_fixed = ct_T
        ct_0 = ct_fixed
        xray_0 = xray_t
        return torch.clip(xray_0, -1, 1), torch.clip(ct_0, -1, 1)
