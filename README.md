# MHSampler
This package aims to generate samples using The Metropolis–Hastings algorithm

`mh(model, priorPDF, likelihood_dist, data, param_dims; proposalPDF=priorPDF, itr = 1000)`


### Inputs

- model 			: Function to generate likelihood value, eg: `model(x) = 3*x+4`
- priorPDF			: Probability density function of prior distribution, eg: Normal(0.0,1.0)
- likelihood_dist	: Distribution of likelihood value, eg: Normal
- data				: Data
- param_dims		: Dimension of paramters

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
	return x.*4 .+3
end;

#PDF of prior distribution
priorPDF = Uniform(0.0,10.0);

likelihood_dist = Normal;

param_dims = 1;
#data generation 

test_params = rand([1.0,2.0,3.0],param_dims)
data = model(test_params);

#sampling
states = mh(model, priorPDF, likelihood_dist, data, param_dims, itr =10000);
histogram(Array(states[1,2:end]), bins=100)
```