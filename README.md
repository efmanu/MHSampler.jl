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
using DataFrames
using Plots
using MHSampler

fo(x, ps1, ps2) = ps1[1].*x +ps1[2].*(x.^2) .+ ps2



input = rand(5)
ps1= (1.0, 2.0)
ps2 = 2.0
ps= (ps1, ps2)
output = fo(input, ps1, ps2)
itr = 10000


proposal_1 = Uniform(0.0,10.0)
proposal_2 = Uniform(0.0,10.0)
length_ps = (length(ps1),length(ps2))
proposals = (proposal_1, proposal_2)

prior_1 = Normal(0.0,2.0)
prior_2 = Normal(0.0,8.0)
priors = (prior_1, prior_2)

model(x, ps1, ps2) = Normal.(fo(x, ps1, ps2), 1.0)

chm = mh(input, output, model, priors, length_ps, itr = itr);
histogram(Array(chm[1,2:end]),  title="MH", bins = 50)
```