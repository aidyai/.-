

Diffussion models are the models powering the most Famous AI Models in the couple of months: IMAGEN, GLIDE, PVD, DDPM, GUIDED DIFFUSION, IMAGEN, DALLE 2

Diffusion models fall into Generative Deep Learning:
This generative models aims to learn the natural distrbution from a given data in order to generate new data. the noise that was added

**PROPERTY**: The latent state has the same dimensionality as the input data
**OPTIMIZATION**: The task is to predict the noise that was added in each of the images 

In order to generate new data one can simply perform the backward process from random noise and new data points are constructed.

This sequential process has t=1000 steps, consequently the larger this number is the slower the sampling will be



#### Generative Adeversarial Networks already exists as well as Variational Auto Encoders and Normalizing Flows

##### 1. DIFFUSION: State of The Art Deep Generatative Model taht produce high quality samples taht are also diverse, beacuse of its sequential nature nature it is quite slow in sampling
##### 2. GANS: They produce high quality outputs but are difficult to train as a result of its mode collapse of vanishing gradients due to its adversarial setup
##### 3. VAES/NF: This tends to produce diverse samples quickly but have issues with bad quality 
##### 4. VAES: What is it?
##### 5. MARKOV CHAIN: A sequence of stochastic events where each time steps depends on the previous time step.
**6. GAUSSIANS:** The sum of gaussians is still a gaussian distribution 



THE DIFFUSION CODE IMPLEMENTATION

**A:** **NOISE SCHEDULER (FORWARD PROCESS):** A schedular that sequentially adds Noise
		This is used in the **FORWARD PROCESS** to add Noise to the Images
						q(x_1:T || x_0) = T_N_t=1  q(x_t||x_t-1)
						q(x_t||x_t-1) = N(x_t;)
		Defining Notations:
					**q:** Denotes the forward process
					**x_0:** This denotes the original input
				**x_t-1:** Previous less noisy image
			**q(x_t || x_t-1) = N(x_t; ...):** Noise sampling formular
		

						
The forward process is denoted by **q** and the noise added to an image depends on the previous image.
The Noise sampler is a conditional gaussian distribution with mean that depends on a previous image and a Variance
Smpling noise from a Diffusion Model

**B:** **NEURAL NETWORK (UNET: BACKWARD PROCESS):** 
A model that predicts the Noise, meaning the UNET Model will take in a noisy 3 colour channel image as input and predict the noise in the image. The model also needs to know which timestep it's in.
	Originally, UNET is one the mainly used models for semantic segmentation, as such the output from the model is the same as the input.
		The input passes through convolution and downsampling layers until it reaches a bottleneck then it gets through convolution and upsampling layers again.
		The input gets more smaller and deeper because more channels are being added.
		The UNET model consists of:
			**a).** Residual connections between layers
			**b).** Batch or group normalization
			**c).** Attention modules
	The reverese or backward process can also be formulated mathemetically like this:
			**p_o(x_t) = N(x_t; 0,1)**:  x_t with gaussian noise with mean of 0 and unit variance 
			formular
	During training timesteps are randomly sampled beacuse we dont want to go through the whole sequence

			**C. OPTIMIZATION (LOSS FUNCTION):** 
			**D. SAMPLING:**
			**E. TRAINING:**

**C:** **TIME STEP ENCODING:**  A way of encoding current timestep into the Model

	







**D:** **NOISE SCHEDULER:** 
