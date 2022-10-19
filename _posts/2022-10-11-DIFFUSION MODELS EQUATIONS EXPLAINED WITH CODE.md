# BASIC DIFFUSION MODELS NOTATIONS

| Notation | Description|
| ----| ----- |
|$T$  | Total number of time steps of the diffusion process |
| $t$ | Time step $t$ on the range of $[0,T]$ |
| $t$ | $[0,T]$ |
| ${\|\cdot\|}$ | ${L_2}$ norm |
| $\mu$ & $\Sigma$ | Mean and Variance |
| $b$ | Bias term |
| $\epsilon$ | Standard Gaussian Noise |
| $x_T$ | Input data becomes indistinguishable from an Isotropic Gaussian Noise |
| $\mathcal{N}$ | Normal Distribution |
| $\beta_t$ | Variance coefficient at time $t$ |
| $\alpha_t$| $1-\beta_t$|
| $\bar{\alpha}{_t}$|Cumulative product of $\alpha_t$ |
| $x$ | Input Data|
| $x_0$ | Unperturbed data in diffusion model |
| $x_t$ | Diffused data in diffusion model  |
| $q({x_t\mid{x_{t-1}}})$| The forward noising Process |
|$q({x_{t-1}\mid{x_t}})$| The backward noising process|
| $\mu_{\theta}({{x_t,t})}$ |Learnable Mean in the backward process at time $t$ |
| $\Sigma_{\theta}(x_t,t)$ | Learnable Variance in the backward process at time $t$ |
|$q(x_t\mid x_{t-1}) = \mathcal{N(x_t; \sqrt{1-\beta_t{x_{t-1}}\beta_t}I)}$|   |
| $L_{(VLB)}$ | Variational Lower Bound|
| $D_{KL}{\space}q(x_T{\mid}x_0){\mid}p(x_T ))$ | Kulliback Leibler Divergence between two Gaussian Distributions |
|$q({x_1\space,{\dots,}\space{x_T}}\mid{x_0})$ |Joint distribution of all the samples generated in the forward process consitioned on $x_0$|






|EQUATION| DESCRIPTION                     |   
|--------|---------------------------------|
|$qr$|<pre>model = python <br> {<br>torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True) ``` </pre>
|$D_{KL}{\space}q(x_T{\mid}x_0){\mid}p(x_T ))$|<pre>def predict_start_from_noise(self, x_t, t, noise):<br>    return( <br>     extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -<br>   extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)</pre>

<table>
<tr>
<td> QWERT </td> <td> EWWEE </td>
</tr>
<tr>
<td>

|```swift
struct: Hello{
public var test: String = "World" //
original
}||
```
</td>
<td>
