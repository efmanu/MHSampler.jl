# MHSampler
This package aims to generate samples using The Metropolis–Hastings algorithm

`mh(model, priorPDF, likelihood_dist, data; proposalPDF=priorPDF, itr = 1000)`


### Inputs

- model 			: Function to generate likelihood value, eg: `model(x) = 3*x+4`
- priorPDF			: Probability density function of prior distribution, eg: Normal(0.0,1.0)
- likelihood_dist	: Distribution of likelihood value, eg: Normal
- data				: Data

### Keyword Arguments
- proposalPDF		: Probability density function to generate proposals, default will be silira to prior PDF. Eg: Normal(0.0,1.0)
- itr 				: Number of samples to generate. Default is 1000.

### Output
- states			: Posterior samples

### Example

```julia
using MHSampler
using Plots

#model to generate likelihood values
function model(x)
	return sum(x)
end;

#PDF of prior distribution
priorPDF = Uniform(0.0,10.0);

likelihood_dist = Normal;

param_dims = 5;
#data generation
data = model(1:param_dims);

#sampling
states = mh(model, priorPDF, likelihood_dist, data, param_dims, itr =10000);
histogram(states, bins=100)
```