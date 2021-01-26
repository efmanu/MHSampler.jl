module MHSampler

using Distributions
using Random
using DataFrames



###########
# Exports #
###########
export mh

"""
	mh(model, priors, length_ps;input = input, output = output,
	proposals = proposals, itr = 1000, burn_in = Int(itr*0.2))
)	
This package aims to generate samples using The Metropolis–Hastings algorithm

# Inputs
- input				: input data
- output			: output data
- model 			: Likelihood distribution, eg: model(x, params) = Normal(f(x,params), 1.0)
- priors			: Prior distribution, eg: Normal(0.0,1.0)
- length_ps			: Length of parameter

# Keyword Arguments
- proposals 		: Proposal distribution, eg: Normal(0.0,1.0)
- itr 				: Number of samples to generate. Default is 1000.
- burn_in 			: To remove warmup samples in the begining

# Output
- states			: Posterior samples

#Example

using Random
using Distributions
using DataFrames
using Plots
using MHSampler


fo1(ps1, ps2) = ps1[1].*ps2 .+ ps1[2]



input = rand(5)
po1= (1.0, 2.0)
po2 = 3.0
ps= (po1, po2)

output = fo1(po1, po2)
itr = 10000

proposal_1 = Uniform(0.0,10.0)
proposal_2 = Uniform(0.0,10.0)
length_ps = (length(po1),length(po2))
proposals = (proposal_1, proposal_2)

prior_1 = Normal(0.0,10.0)
prior_2 = Normal(2.0,8.0)
priors = (prior_1, prior_2)


model(x, ps1, ps2) = Normal.(fo1(ps1, ps2), 1.0)

chm = mh(priors, length_ps, model = model, input = input, output=output, itr = itr, burn_in = 1);
histogram(Array(chm[1,2:end]),  title="MH", bins = 50)
"""

function mh(priors, length_ps;
	model = nothing, 
	input = Array{Float64}(undef,0), 
	output = Array{Float64}(undef,0),
	proposals = priors, itr = 1000, burn_in = Int(itr*0.2)
)	
	states = Dict()
	function logJoint(params)	
		logPrior= sum(map(logParams, priors, params))
		logLikelihood = 0.0
		if !(model isa Nothing)
			logLikelihood = sum(logpdf.(model(input, params...), output))
		end
		return logPrior + logLikelihood
	end

	prev_params = map(rand, proposals, length_ps)
	logJoint_prevs = logJoint(prev_params)
	states["itr_1"] = prev_params
	for i in 2:itr	
		current_params = map(rand, proposals, length_ps)	
		logJoint_curs = logJoint(current_params)
		logα = logJoint_curs - logJoint_prevs
		if (-Random.randexp() < logα)						
			prev_params = current_params
			logJoint_prevs = logJoint_curs
		end
		states["itr_$i"] = prev_params
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
		chain[!,Symbol((i-burn_in))] = rand(sum(length_ps));
		jk = 0
		for ln in 1:lps			
			for x in 1:length_ps[ln]
				jk += 1
				chain[jk,(i-burn_in)+1]= states["itr_$i"][ln][x]
			end
		end
	end
	return chain
end
end # module
