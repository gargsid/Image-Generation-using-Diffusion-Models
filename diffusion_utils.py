import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from configs import sm_butterfly_data_config as config 

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

image_size = config['model_params']['image_size']

# DDPM noise schedule 
class DDPMScheduler:
    def __init__(self, timesteps, beta1, beta2):
        # construct DDPM noise schedule
        self.timesteps = timesteps
        self.beta2 = beta2
        self.beta1 = beta1

        self.b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0] = 1

    # function to add noise to input during training
    def perturb_input(self, x, t, noise):
        return self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]) * noise

    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise

    @torch.no_grad()
    def sample_ddpm(self, n_sample, model, save_rate=20, context=None):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, image_size, image_size).to(device)  
        
        # array to keep track of generated steps for plotting
        intermediate = [] 
        timesteps = config['diffusion_params']['timesteps']
        for i in range(timesteps, 0, -1):
            # print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = model(samples, t, context)    # predict noise e_(x_t,t)
            samples = self.denoise_add_noise(samples, i, eps, z)
            # print(i, save_rate, timesteps)
            if i % save_rate ==0 or i==timesteps or i<8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate

    # removes the noise using ddim
    def denoise_ddim(self, x, t, t_prev, pred_noise):
        ab = self.ab_t[t]
        ab_prev = self.ab_t[t_prev]
        
        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt

    # define sampling function for DDIM   
    @torch.no_grad()
    def sample_ddim(self, n_sample, model, n=20):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, image_size, image_size).to(device)  

        # array to keep track of generated steps for plotting
        intermediate = [] 
        step_size = timesteps // n
        for i in range(timesteps, 0, -step_size):
            # print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            eps = model(samples, t)    # predict noise e_(x_t,t)
            samples = denoise_ddim(samples, i, i - step_size, eps)
            intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate


# # TO_DO
# class CosineNoiseScheduler:
#     def __init__(self, timesteps,)
