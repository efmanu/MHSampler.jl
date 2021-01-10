module MHSampler

using Distributions
using Random
using DataFrames

###########
# Exports #
###########
export mh

"""
	mh(model, priorPDF, likelihood_dist, data, param_dims; proposalPDF=priorPDF, itr = 1000)
This package aims to generate samples using The Metropolis–Hastings algorithm

# Inputs
- input				: input data
- output			: output data
- model 			: Likelihood distribution, eg: model(x, params) = Normal(f(x,params), 1.0)
- prior				: Prior distribution, eg: Normal(0.0,1.0)
- length_ps			: Length of parameter

# Keyword Arguments
- proposal 			: Proposal distribution, eg: Normal(0.0,1.0)
- itr 				: Number of samples to generate. Default is 1000.

# Output
- states			: Posterior samples

#Example

using Random
using Distributions
using DataFrames
using Plots
using MHSampler

f(x, ps) = ps[1].*x .+ ps[2]

model(x, ps) = Normal.(f(x, ps), 1.0)

input = rand(5)
ps = [1,2]
output = f(input,ps)

length_ps = 2

prior = Uniform(0.0,10.0)
proposal = Uniform(0.0,10.0)

itr = 10000
ch = mh(input, output, model, prior, length_ps)
histogram(Array(ch[1,2:end]), bins = 50)
"""
function mh(input, output, model, prior, length_ps; proposal = prior, itr = 1000, burn_in = Int(itr*0.2))
	states = DataFrame();
	states.var = map(x->"param[$x]", 1:length_ps)

	function logJoint(params)
		psn = rand(proposal, length_ps)
		logPrior = sum(logpdf.(prior, psn))
		logLikelihood = sum(logpdf.(model(input, psn), output))
		return logPrior + logLikelihood
	end

	prev_params = rand(proposal, length_ps)
	logJoint_prev = logJoint(prev_params)

	for i in 2:itr	
		states[!,Symbol(i-1)] = prev_params	
		current_params = rand(proposal, length_ps)		
		logJoint_cur = logJoint(current_params)
		logα = logJoint_cur - logJoint_prev
		if (-Random.randexp() < logα)
			prev_params = current_params
			logJoint_prev= logJoint_cur
		end
	end
	for i in 1:burn_in
		select!(states,Not(Symbol(i)))
	end
	return states
end


end # module
