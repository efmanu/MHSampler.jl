module MHSampler

using Distributions
using Random

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
function mh(model, priorPDF, likelihood_dist, data; proposalPDF=priorPDF, itr = 1000)
	states = Array{Float64}(undef,0)
	burn_in = Int(itr*0.2)
	#initial value
	prev_params = 1.0
	for i=2:itr
		append!(states,prev_params)
		params = rand(proposalPDF)
		lg_curr = log_joint(model,priorPDF, params, likelihood_dist, data)	 
		lg_prev = log_joint(model,priorPDF, prev_params, likelihood_dist, data)	 

		logα = lg_curr - lg_prev

		#acceptance criteria
		if(-Random.randexp() < logα)
			prev_params = params
		end
	end
	return states[burn_in:itr-1]
end
"""
	log_joint(model,priorPDF, params, likelihood_dist, data)	
Function identify log of joint distribution
"""
function log_joint(model,priorPDF, params, likelihood_dist, data)	 
	logpdf_prior = pdf(priorPDF, params)
	pred = model(params)
	likelihoodPDF = likelihood_dist(pred, 1.0)
	logpdf_likelihood = logpdf(likelihoodPDF, data)
	return (logpdf_likelihood + logpdf_prior)
end

end # module
