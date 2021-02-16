# MHSampler
This package aims to generate samples using The Metropolis–Hastings algorithm

```julia
	mh(priors, proposals::Function;
		model = nothing, 
		output = Array{Float64}(undef,0),
		itr = 1000, burn_in = Int(itr*0.2)
	)	
```

### Inputs
- priors			: Prior distribution, eg: Normal(0.0,1.0)
- proposals			: proposals is proposal generating function. Eg: proposals() = rand(Normal(0.0,1.0))

### Keyword Arguments
- output			: output data
- model 			: Likelihood distribution, eg: model(x, params) = Normal(f(x,params), 1.0)
- itr 				: Number of samples to generate. Default is 1000.
- burn_in 			: To remove warmup samples in the begining

### Output
- chain				: Posterior samples

### Example

```julia

#Example

using Distributions
using Plots
using MHSampler

l_w = 10
M = 3
W_mean = rand(l_w)
P = rand(l_w,M)
z = rand(M)
input = rand(l_w)
output = rand(l_w)
model(prm) = Normal.((input.*prm), 5.0)

proposald = MvNormal(ones(l_w), 3.0)
prior = MvNormal(ones(l_w), 3.0)
proposalf() = rand(proposald)

chm = mh(prior, proposalf, model=model, output = output, itr =10_000)

histogram(Array(chm[1,2:end]),  title="MH", bins = 50)
```