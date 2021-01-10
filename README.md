# MHSampler
This package aims to generate samples using The Metropolis–Hastings algorithm

`mh(input, output, model, prior, length_ps; proposal = prior, itr = 1000, burn_in = Int(itr*0.2))`

### Inputs
- input				: input data
- output			: output data
- model 			: Likelihood distribution, eg: model(x, params) = Normal(f(x,params), 1.0)
- prior				: Prior distribution, eg: Normal(0.0,1.0)
- length_ps			: Length of parameter

### Keyword Arguments
- proposal 			: Proposal distribution, eg: Normal(0.0,1.0)
- itr 				: Number of samples to generate. Default is 1000.

### Output
- states			: Posterior samples

### Example

```julia
using Random
using Distributions
using DataFrame
using Plots

f(x, ps) = ps[1].*x .+ ps[2]

model(x, ps) = Normal.(f(x, ps), 1.0)

input = rand()
ps = [1,2]
output = f(input,ps)

length_ps = 2

prior = Uniform(0.0,10.0)
proposal = Uniform(0.0,10.0)

itr = 1000
ch = mh(input, output, model, prior, length_ps)
histogram(Array(ch[1,2:end]))
```