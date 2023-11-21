import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
from tqdm import tqdm 
import os

from sprite_data import CustomDataset
from unet import ContextUnet
from config import *

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ddpm')
parser.add_argument('--context', action='store_true')

timesteps = diffusion_params['timesteps']
beta2 = diffusion_params['beta2']
beta1 = diffusion_params['beta1']

b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1

def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0,1))
    xmin = x.min((0,1))
    return(x - xmin)/(xmax - xmin)

def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t,s] = unorm(store[t,s])
    return nstore

def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn,  w, save=False):
    ncols = n_sample//nrows
    sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
    
    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
        return plots
    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0]) 
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
    return ani

def show_images(imgs, fname, nrow=2):
    _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4,2 ))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
    plt.savefig(f'{save_dir}/{fname}.png')
    print(f'fig saved to {save_dir}/{fname}.png')

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample, model, save_rate=20, context=None):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, hyperparams['image_size'], hyperparams['image_size']).to(device)  
    
    # array to keep track of generated steps for plotting
    intermediate = [] 
    timesteps = hyperparams['timesteps']
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = model(samples, t, context)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        # print(i, save_rate, timesteps)
        if i % save_rate ==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

# define sampling function for DDIM   
# removes the noise using ddim
def denoise_ddim(x, t, t_prev, pred_noise):
    ab = ab_t[t]
    ab_prev = ab_t[t_prev]
    
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
    dir_xt = (1 - ab_prev).sqrt() * pred_noise

    return x0_pred + dir_xt

@torch.no_grad()
def sample_ddim(n_sample, model, n=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, image_size, image_size).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    step_size = timesteps // n
    for i in range(timesteps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        eps = model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, i - step_size, eps)
        intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

hidden_size = model_params['hidden_size']
num_classes = model_params['num_classes']
image_size = model_params['image_size']

if args.method == 'ddpm':
    save_dir = ddpm['save_dir']
elif args.method == 'ddim':
    save_dir = ddim['save_dir']

if args.context:
    save_dir += 'context/'
else:
    save_dir += 'no_context/'
    
save_dir += 'ckpt/'

model = ContextUnet(3, hidden_size, num_classes, image_size).to(device)

model.load_state_dict(torch.load(f"{save_dir}/model_31.pth"))
model.eval()
print("Loaded in Model")

plt.clf()
samples, intermediate = sample_ddim(32, model, n=25)
animation_ddim = plot_sample(intermediate,32,4,save_dir, "ani_run", '_ddim', save=True)


# plt.clf()
# ctx = torch.tensor([
#     # hero, non-hero, food, spell, side-facing
#     [1,0,0,0,0],  
#     [1,0,0,0,0],    
#     # [0,1,0,0,0],
#     # [0,1,0,0,0],
#     [0,0,1,0,0],
#     [0,0,1,0,0],
#     # [0,0,0,1,0],
#     # [0,0,0,1,0],    
#     # [0,0,0,0,1],
#     # [0,0,0,0,1],
# ]).float().to(device)
# samples, _ = sample_ddpm(ctx.shape[0], model, context=ctx)
# show_images(samples, 'ddpm_context')
# samples, intermediate_ddpm = sample_ddpm(32, model)
# animation_ddpm = plot_sample(intermediate_ddpm, 32, 4, save_dir, "ani_run", None, save=True)
# HTML(animation_ddpm.to_jshtml())