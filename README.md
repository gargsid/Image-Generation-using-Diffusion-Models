# Image Generation Using Diffusion Models

This repository implements Diffusion Models using methods described in these papers -- [DDPM](https://arxiv.org/pdf/2006.11239.pdf) and [DDIM](https://arxiv.org/pdf/2010.02502.pdf). We use a basic version of UNet2D architecture for generating sprite images which are 16x16 in size. We train the diffusion models in the pixel space using and use Linear Noise Schedule during the forward process. We compare two sampling techniques presented in DDPM and DDIM papers. We showed that DDIM method can improve the sampling speed when compared with DDPM method but with a trade-off in image qualities. 

## Results

Sample generation using DDPM

<img src="https://github.com/gargsid/Image-Generation-using-Diffusion-Models/blob/main/assets/ani_run_wNone.gif" width="1200" height="500" />

Sample generation using DDIM

<img src="https://github.com/gargsid/Image-Generation-using-Diffusion-Models/blob/main/assets/ani_run_w_ddim.gif" width="1200" height="500" />


## Acknowledgements

DeepLearning.AI Course: [How Diffusion Models Work?](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)

