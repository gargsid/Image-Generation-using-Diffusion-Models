import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt

import argparse
from tqdm import tqdm 
import os

from data_utils import SpriteDataset, SmithsonianButterflies
from unet import ContextUnet
from configs import sm_butterfly_data_config as config
from diffusion_utils import DDPMScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ddpm')
parser.add_argument('--context', action='store_true')
parser.add_argument('--testing', action='store_true')
args = parser.parse_args()

batch_size = config['training_params']['batch_size']
lrate = config['training_params']['learning_rate']
image_size = config['model_params']['image_size']

dataset = SmithsonianButterflies(image_size, context=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)

hidden_size = config['model_params']['hidden_size']
num_classes = dataset.num_classes
# print('num_classes:', num_classes)

# model = ContextUnet(3, hidden_size, num_classes, image_size).to(device)
# image_size=128
down_block_channels = config['model_params']['down_block_channels']
model = UNet2DModel(in_channels, down_block_channels, image_size).to(device)

optim = torch.optim.Adam(model.parameters(), lr=lrate)
n_epoch = config['training_params']['n_epoch'] 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epoch)

# construct DDPM noise schedule
timesteps = config['diffusion_params']['timesteps']
beta2 = config['diffusion_params']['beta2']
beta1 = config['diffusion_params']['beta1']

ddpm_scheduler = DDPMScheduler(timesteps, beta1, beta2)

# set into train mode
model.train()



if args.method == 'ddpm':
    save_dir = config['ddpm']['save_dir']
elif args.method == 'ddim':
    save_dir = config['ddim']['save_dir']

if args.context:
    save_dir += 'context/'
else:
    save_dir += 'no_context/'
    
save_model_dir = save_dir + 'ckpt/'
save_image_dir = save_dir + 'images/'

os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_image_dir, exist_ok=True)

# set into train mode
model.train()

n_epoch = config['training_params']['n_epoch']

for ep in range(n_epoch):
    print(f'epoch {ep+1}/{n_epoch}', end=' ')
    
    # linearly decay learning rate
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

    total_loss = 0.
    
    # pbar = tqdm(dataloader, mininterval=2 )
    # for x, c in pbar:   # x: images
    for x, c in dataloader:   # x: images
        optim.zero_grad()
        x = x.to(device)

        if args.context:
            c = c.to(device) # added code
            # randomly mask out c
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)
        else:
            c = None

        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
        x_pert = ddpm_scheduler.perturb_input(x, t, noise)
        
        # use network to recover noise
        pred_noise = model(x_pert, t / timesteps, c=c)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        total_loss += loss.item()
        loss.backward()
        
        optim.step()

        if args.testing:
            break
    total_loss/=len(dataloader)
    scheduler.step()

    print('total_loss:', total_loss)

    # save model periodically
    if (ep+1) % 50 == 0 or ep == int(n_epoch-1) or args.testing:
        # saving the ckpt
        torch.save(model.state_dict(), save_model_dir + f"model_{ep+1}.pth")
        print('saved model at ' + save_model_dir + f"model_{ep+1}.pth")

        # sampling some images 
        if not args.context:
            context = None
        else:
            context = args.context
        samples, _ = ddpm_scheduler.sample_ddpm(8, model, context=context)

        nrow=2
        _, axs = plt.subplots(nrow, samples.shape[0] // nrow, figsize=(4,2 ))
        axs = axs.flatten()
        for img, ax in zip(samples, axs):
            img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img)
        plt.savefig(f'{save_image_dir}/{args.method}_ep_{ep+1}.png')
        print(f'fig saved to {save_image_dir}/{args.method}_ep_{ep+1}.png')
        plt.close()

    if args.testing:
        break
