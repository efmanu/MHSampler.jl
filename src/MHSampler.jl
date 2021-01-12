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
- priors			: Prior distribution, eg: Normal(0.0,1.0)
- length_ps			: Length of parameter

# Keyword Arguments
- proposals 			: Proposal distribution, eg: Normal(0.0,1.0)
- itr 				: Number of samples to generate. Default is 1000.

# Output
- states			: Posterior samples

#Example

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

"""

function mh(input, output, model, priors, length_ps; proposals = priors, itr = 1000, burn_in = Int(itr*0.2))
	# states = DataFrame();
	# states.var = map(x->"param[$x]", 1:length_ps)
	states = Dict()
	function logJoint(params)	
		logPrior= sum(map(logParams, priors, params))
		logLikelihood = sum(logpdf.(model(input, params...), output))
		return logPrior + logLikelihood
	end

	prev_params = map(rand, proposals, length_ps)
	logJoint_prevs = logJoint(prev_params)

	for i in 2:itr	
		# states[!,Symbol(i-1)] = prev_params
		states["itr_$(i-1)"] = prev_params
		current_params = map(rand, proposals, length_ps)	
		logJoint_curs = logJoint(current_params)
		logα = logJoint_curs - logJoint_prevs
		if (-Random.randexp() < logα)						
			prev_params = current_params
			logJoint_prevs = logJoint_curs
		end
	end
	
	return data_formatting(states, length_ps, burn_in, itr)
end

logParams(prior, params) = sum(logpdf.(prior, params))

function data_formatting(states, length_ps, burn_in, itr)
	chain =DataFrame();
	lps = length(length_ps)
	param_names =[]
	for ln in 1:lps
		for x in 1:length_ps[ln]
			push!(param_names,"param[$ln]_[$x]")
		end
	end
	chain.var = param_names
	for i in (burn_in+1):itr
		# bt = i-burn_in
		chain[!,Symbol((i-burn_in))] = rand(sum(length_ps));
		for ln in 1:lps
			for x in 1:length_ps[ln]
				chain[(ln-1)+x,(i-burn_in)+1]= states["itr_$(i)"][ln][x]
			end
		end
	end
	return chain
end
end # module
