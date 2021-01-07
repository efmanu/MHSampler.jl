module MHSampler

using Distributions
using Random

"""
	mh(η)
To generate samples

# Inputs
- `propPDF`: Proposal distribution
- `priorPDF`: Prior distribution
# Keyword Arguments
- `η`	: Acceptance level
- `itr` : Number iterations to generate samples
"""
function mh(model, priorPDF, likelihood_dist, data; proposalPDF=priorPDF, η = 0.65, itr = 1000)
	states = Array{Float64}(undef,0)
	burn_in = Int(itr*0.2)
	prev_params = 1.0
	for i=1:itr
		append!(states,prev_params)
		params = rand(proposalPDF)
		lg_curr = log_joint(model,priorPDF, params, likelihood_dist, data)	 
		lg_prev = log_joint(model,priorPDF, prev_params, likelihood_dist, data)	 

		logα = lg_curr - lg_prev
		if(-Random.randexp() < logα)
			prev_params = params
		end
	end
	return states[burn_in:itr-1]
end
function log_joint(model,priorPDF, params, likelihood_dist, data)	 
	logpdf_prior = pdf(priorPDF, params)
	pred = model(params)
	likelihoodPDF = likelihood_dist(pred, 1.0)
	logpdf_likelihood = logpdf(likelihoodPDF, data)
	return (logpdf_likelihood + logpdf_prior)
end

end # module
