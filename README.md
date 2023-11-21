# Image Generation Using Diffusion Models

This repository implements Diffusion Models called DDPM and DDIM. We use a basic UNet2D architecture for generating sprite images which are 16x16 in size. We train the diffusion models in the pixel space using and use Linear Noise Schedule during the forward process. We compare two sampling techniques presented in DDPM and DDIM papers. We showed that DDIM method can improve the sampling speed when compared with DDPM method but with a trade-off in image qualities. 

## Results

Sample generation using DDPM



## Acknowledgements

DeepLearning.AI Course: (https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)[How Diffusion Models Work?]

