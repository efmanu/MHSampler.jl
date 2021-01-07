module MHSampler

using Distributions
using Random
using DataFrames

###########
# Exports #
###########
export mh

"""
	mh(model, priorPDF, likelihood_dist, data; proposalPDF=priorPDF, itr = 1000)
This package aims to generate samples using The Metropolis–Hastings algorithm

# Inputs
- model 			: Function to generate likelihood value, eg: model(x) = 3*x+4
- priorPDF			: Probability density function of prior distribution, eg: Normal(0.0,1.0)
- likelihood_dist	: Distribution of likelihood value, eg: Normal
- data				: Data

# Keyword Arguments
- proposalPDF		: Probability density function to generate proposals, default will be silira to prior PDF. Eg: Normal(0.0,1.0)
- itr 				: Number of samples to generate. Default is 1000.

# Output
- states			: Posterior samples

#Example

states = mh(model, priorPDF, likelihood_dist, data)
"""
function mh(model, priorPDF, likelihood_dist, data, param_dims; proposalPDF=priorPDF, itr = 1000)
	states = DataFrame();
	states.var = map(x->"var[$x]", 1:param_dims)
	burn_in = Int(itr*0.2)
	#initial value
	prev_params = ones(param_dims)
	for i=2:itr
		states[!,Symbol(i-1)] = rand(param_dims)
		params = rand(proposalPDF, param_dims)
		lg_curr = log_joint(model,priorPDF, params, likelihood_dist, data)	 
		lg_prev = log_joint(model,priorPDF, prev_params, likelihood_dist, data)	 

		logα = lg_curr - lg_prev

		#acceptance criteria
		if(-Random.randexp() < logα)
			prev_params = params
		end
	end
	for i  in 1:burn_in
		select!(states,Not(Symbol(i)))
	end
	return states
end
"""
	log_joint(model,priorPDF, params, likelihood_dist, data)	
Function identify log of joint distribution
"""
function log_joint(model,priorPDF, params, likelihood_dist, data)
	logpdf_prior = map(x->pdf(priorPDF,x), params)
	pred = model(params)
	likelihoodPDF = map(x->likelihood_dist(x,1.0), pred)
	logpdf_likelihood = logpdf.(likelihoodPDF, data)
	return (sum(logpdf_likelihood) + sum(logpdf_prior))
end


end # module
