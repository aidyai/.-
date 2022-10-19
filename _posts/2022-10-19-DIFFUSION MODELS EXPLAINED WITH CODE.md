# DIFFUSION MODELS
#### INTRO______

It is very obvious that Diffusion model have taken over GANS as the State of The Art Generative model in the Computer Vision and Multi-Modal Learning corridors.

In this 3 Part series tutorial we will learn about Diffusion Models, How it works with pytorch code added, The Various Architectures, and the some of it Applications.
Then we will work through an End to End App that can take a low quality image like a sketch and generate a high fidelity one


### INTODUCTION
DIffusion model are a class of state-of-the art deep generative models that have shown impressive results on various tasks ranging from multi-modal modelling Vision to NLP to waveform signal Processing.
___
#### GENERATIVE MODELS
What then is a Generative Model? 
It is a model used to generate data similar to the data which it was trained on. A generative model attempts to learn the data distribution $p(x|y)$.

What is agenerative model?.............................................-----

Onced trained, a generative model can generate novel samples from an approximation of $p(x|y)$, denoted $p_\theta({x|y})$ 
___

#### DENOISING DIFFUSION PROBABILISTIC MODELS
Diffusion models have achieved very impressive quality and diversity of sample synthesis than other state-of-the art genrative models.

Though with problems, they suffer from slow sampling,... and ...
___
##### A. NOISING:
Diffusion models are a latent variable generative model inspired by non-equilibrum thermodynamics from physics.  and generation is modeled as a denoising process. It works by adding gaussian noise to original image following a Markov Chain. It is basically a Markov chain where in each time step we add a little bit of gaussian noise to an image untill it is completely destroyed by gaussian noise.
___
###### What It Means to Add Gaussian Noise?
Firstly, gaussian noise is 


____
##### B. DENOISING:
Starting from sampled noise, the diffusion model performs $T$ denoising steps until a sharp image is formed. 

The denosing process produces a series of intermediate images with decreasing levels of noise, denote as $x_T, x_{T-1},...x_0$,

Given only $x_T$ which is indistinguishable from gaussian noise we can get $x_0$ an output image. 

#### What is Means to Reverse The Noise?
Reversing or removing the noise means recovering the values of the pixels, so that the resulting image will be similar the original image.


___
This class of models consist of two processes:
1. The first is a forward process that follows a Markov Chain which progrssively disturbs the data distribution by adding gaussian noise.

2. The second is a reverse process that learns to restore the data structure

___
This forward and backward Markov Chain process uses Variational inference to produce samples matching the original data after a finite time. 
___

___
1. The forward chain pertubs the data distribution by gradually adding Gaussian noise to the ground truth image with a pre-designed schedule until the data distribution converges to a given prior, i.e., a standard Gaussian Distribution -- (Isometric Gaussian).
$$ q(x_1,...,x_T|x_0) = \prod^T_{t=1}   {q(x_t|x_{t-1})} ---(1)$$
$$ q(x_t|x_{t-1}) = \mathcal{N(x_t; \sqrt{1-\beta_t{x_{t-1}}\beta_t}I)}   ---(2)$$ 
$q(x_t)$ is used to denote the distributions of latent variables $x_t$ in the forward process.

The noising process defined in Eq.(2) allows us to sample an arbitrary step of the noised latents directly conditioned on the input $x_o$.
	Where $\alpha_t$ = $1-\beta_t$ and $\bar{\alpha_t}$ = $\prod^t_{s=0}$ $\bar{\alpha_s}$, we can wite the marginal as:
	$$ q(x_t|x_0) = \mathcal{N(x_t;\sqrt{\bar{\alpha_t}}x_0,(1-{\bar{\alpha_t}})I)} $$
$$ x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1-{\bar{\alpha_t}}}\epsilon $$
When $\bar{\alpha}_t$ approximates 0, $x_T$ is practically indistinguisahble from pure Gaussian noise: $p(x_T)\approx$ $\mathcal{N}(x_T;0,1)$.
___
2. The reverse process takes the completely noised image and learns to gradually revert the Markov chain of noise corruption to the ground truth. The reversed process is then written as follows: 
$$ p_\theta({x_{t-1}|x_t}) = \mathcal{N}(x_{t-1};\mu(x_t,t, {\Sigma}_\theta(x_t,t)) $$
___
In Denoising Diffusion Models,  The Noise $\epsilon$ is what is Predicted and this is done by Optimizing the variational upper bound on the negative log-likelihood.
___
$$
\\E\ [-\log{p_{\theta}}(x_0)]< \\ E_q[-\log\frac{p_\theta({x_{0:T}})}{q(x_{1:T}|x_0)}]--(7) $$
$$ \\ E_q[-\log{p{(x_T)}}-\sum_{t>=1}{\log\frac{p_\theta({x_{0:T}})}{q(x_{1:T}|x_0)}} ]-- (8)$$
$$= -L_{VLB}$$
___
Reparametrization have been appplied to Eq. (8), which results in the general objective below:
$$  E_{t\sim{\mathcal{U(0,T),x_0{\sim{q{(x_0),\epsilon\sim{\mathcal{N(0,1)}}}}}}}}[\lambda{(t)}||\epsilon-{\epsilon_\theta}(x_t,t)||^2]   --(10)$$
___
The neural network ${\epsilon_\theta}(x_t,t)$ predicts $\epsilon$ by minimizing the loss = ${||\epsilon-\epsilon_\theta}(x_t,t)||^2$ which is the $L_2$ Loss 
___
###### INTUITIVELY: 
Given a noised image, the noise added is predicted and then this predicted noise is subtracted from the noise to get the real image. This is basically what is happenning when training a Diffusion Model, It is learning to denoise.
___
#### Some Common Applications of  Diffusion Models
a) Image Super Resolution
b) Image Inpainting
c) Image Outpainting
d) Semantic Segmentation
e) Point Cloud Completion nd Generation
f) Text-to-Image Generation
g) Text-to-Audio Generation 
___









____
##### REFERENCES
1. Karsten Kreis; Ruigui Gao; Arash Vahdat (2022-5-4): "Denoising Diffusion-based Generative Modelling: Foundations and Applications (CVPR 2002 Worskshop) "

___


























___
Both the forward Diffusion processes $q(x_t|x_{t-1})$ and the backward or reconstruction process $q(x_{t-1}|x_t)$ are modelled as the products of Markov transition probabilities:
$$q(x_{0:T}) = q(x_0)\prod_{t=1}^T{q(x_t|x_{t-1})}, p_\theta(x_{T:0}) = p(x_{T})\prod_{t=T}^1{p_{\theta}(x_{t-1}|x_t)},$$
$q(x_0)$ is the real data distribution


### DIFFUSION MODEL
...
...
...
Basic Idea of Diffusion
Forward Diffusion
Reparametrization tricks
Variance Schedule
Reverse Diffusion
Training a Diffusion Model
...

Architecture
Conditional Image generation
	Classifier guidiance
	Classifier free guidiance


CONCLUSION


#### IMPORTant
Diffusion models are latent variable models
	Latent variables$:$ = $x_1,x_{2},x_3,x_4,\cdots x_T$  
	Observed variables$:$ $x_0$
	
	In ddpms the forward process is fixed while the reverse process is what needs learning meaning we need to train only a single neural network

#### TRAINING A DENOSING DIFFUSION PROBABILISTIC MODEL
The reverse step pocess is only tasked with learning the means while its variance is set to a constant
______
##### OBJECTIVE FUNCTION OF A DDPM
$$E_{x_0{\sim{q{(x_0),\epsilon\sim{\mathcal{N(0,1)t\sim{\mathcal{U(0,T),}}}}}}}}[||\epsilon-{\epsilon_\theta}(\sqrt{\bar{\alpha}}\space x_0 + \sqrt{1-\bar{\alpha_t}}\space \epsilon),t||^2]$$
	Where $x_t = \sqrt{\bar{\alpha}}\space x_0 + \sqrt{1-\bar{\alpha_t}}\space \epsilon)$

$E_{x_0{\sim{q{(x_0),\epsilon\sim{\mathcal{N(0,1)t\sim{\mathcal{U(0,T),}}}}}}}}[||\epsilon-{\epsilon_\theta}(x_t,t)||^2]$

Our loss funtion finally looks like this$:$
			$L_{simple}={E_{x_0,t,\epsilon}}[||\epsilon_-\epsilon_{\theta}(x_t,t)||^2]$
________
###### The Training Algorithm looks like this:
![[trainDIFF.PNG]]










#### SAMLING FROM A DENOSING DIFFUSION PROBABILISTIC MODEL














_______
###### The Sampling Algorithm looks like this:
![[sampleDIFF.PNG]]



#### NEURAL NETWORK ARCHITECTURE




















_____

## DENOISING DIFFUSION BASED GENERATIVE MODELS

denoising diffusion models consists of two processes:
1. forward diffusion process that gradually adds noise to input (image)
2. reverse diffusion process that learns to generate data by denoising: it takes a noisy image and learns to generate a less noisy version of that image, this process will be repeated until noise is converted to data


#### DEFINING THE PROCESS FORMALLY
###### FORWARD PROCESS:
The forward diffusion process starts from data (image) and generates this intermediate noisy images by simply adding noise one step at a time
At every step, a normal distribution will be used to generate an image conditioned on the previous image.

the normal distribution which is represented as $q(x_t|x_{t-1}) = \mathcal{N(x_t; \sqrt{1-\beta_t{x_{t-1}}\beta_t}I)}$ is goint to take $x_{t-1}$ the prevoius step and generate $x_t$ the current step. It takes $x_0$ and it generates $x_1$ 

A noraml distribution over the current step $x_t$ where the mean is $\mathcal{\sqrt{1-\beta_t}})$ times the image at the prevoius time step which is ${x_{t-1}}$ and ${\beta_t}I$ represents the variance scheduler which in the real sense is a very small positive scalar value $0.001$  


This normal distribution, $\mathcal{N(x_t; {\sqrt{1-\beta_t}{x_{t-1}}, \beta_t}I)}$  takes the image at the previous step, rescales the pixel values in this image and then adds tiny bit of noise via the variance scheduler "per time step"

###### JOINT DISTRIBUTION:
A joint distibution can also be defined for all the samples generated in the forward process starting from $x_1$ all the way to $x_T$.  The joint distribution which is the samples conditioned on $x_0$ is the cumulateive product of the conditionals that are formed at each step as such $q(x_1,...,x_T|x_0)$ defines the joint distribution of all the samples that will be generated in the forward markov process
$$q(x_1,...,x_T|x_0) = \prod^T_{t=1}   {q(x_t|x_{t-1})}$$
###### Speed?
Why can't we use $x_0$ our input image to generate noisy samples at any time step say $x_{10}$. Simply put can't we use $x_0$ to generate $x_{10}$ ?. 
We can do that by making
${\alpha}_t$ = $1-\beta_t$ , 
then $\bar{\alpha}_t$ which is the cumulative product of ${\alpha}_t$ now becomes
$$\bar{\alpha}_t = \prod^{t}_{s=1}(1-\beta_s)$$
In order to answer the speed question we can then rewrite the original formular as follows:
		$q(x_t|x_0) = \mathcal{N(x_t;\sqrt{\bar{\alpha_t}}x_0,(1-{\bar{\alpha_t}})I)}$
Using the reparameterization trick we can sample $x_t$ as follows
$x_t$ = $\sqrt{\bar{\alpha}}\space x_0 + \sqrt{1-\bar{\alpha_t}}\space \epsilon$  where $\epsilon \sim{\mathcal{N(0,1)}}$  and ${1-\bar{\alpha_t}}$ is our noise schedule at any time step, as such given $x_0$ we can draw samples at any time step $t$. 

It should also be noted that the forward diffusion process is defined such that as  $(x_T\mid{x_0})$ approaches infinity it becomes indistinguishable from standard normal distribution $\mathcal{N({x_T;(0,1)})}$. 
 

#### DENOISING: DEFINING THE GENERATIVE MODEL BY DENOISING

In order to generate data from a diffusion model, we will start from pure noise which is a standard normal distribution with zero mean and unit variance and generates data by denoising one step at a time.
					
					img(denoise)
					
As such $p(x_T)$ = $\mathcal{N(x_T;(0,1))}$  is the distribution of data at the end of the forward diffusion process.
the parametric denoising distribution can be defined as follows $p_\theta(x_{t-1}\mid{x_t})$ = $\mathcal{N({x_{t-1};{\mu}_\theta{(x_t,t)},{\Sigma_\theta({x_t,{t}}})})}$ apart from the sample $x_t$ at time $t$ the model also takes $t$ as input in order to account for the different noise levels at different time steps in the forward process noise schedule so that the model can learn to undo this individually

#### Joint Distribution

The joint distribution can be written as 
It is the product of the base distibution $p{(x_T)}$ and the product of the conditionals which still follows a markov process 

		$x_0\Leftarrow \cdots  \Leftarrow \cdots  \Leftarrow x_{T-1}\cdots  \Leftarrow x_T$

$$p_\theta(x_{0:T})=p(x_T)\prod^T_{t=1}p_\theta(x_{t-1}|x_t)$$






















