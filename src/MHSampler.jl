module MHSampler

using Distributions
using Random
using DataFrames



###########
# Exports #
###########
export mh

"""
	mh(priors, proposals::Function;
	model = nothing,  
	output = Array{Float64}(undef,0),
	itr = 1000, burn_in = Int(itr*0.2)
)		
This package aims to generate samples using The Metropolis–Hastings algorithm

# Inputs
- priors			: Prior distribution, eg: Normal(0.0,1.0)
- proposals			: proposals is proposal generating function. Eg: proposals() = rand(Normal(0.0,1.0))

# Keyword Arguments

- output			: output data
- model 			: Likelihood distribution, eg: model(x, params) = Normal(f(x,params), 1.0)
- itr 				: Number of samples to generate. Default is 1000.
- burn_in 			: To remove warmup samples in the begining

# Output
- chain				: Posterior samples

#Example

using Distributions
using Plots
using MHSampler

l_w = 10
M = 3
W_mean = rand(l_w)
P = rand(l_w,M)
z = rand(M)
input = rand(l_w)
output = rand(l_w)
model(prm) = Normal.((input.*prm), 5.0)

proposald = MvNormal(zeros(l_w), 2.0)
prior = MvNormal(zeros(l_w), 3.0)
proposalf() = rand(proposald)

chm = mh(prior, proposalf, model = model, output = output)

histogram(Array(chm[1,2:end]),  title="MH", bins = 50)
"""


function mh(priors , proposals::Function;
	model = nothing, 
	output = nothing,
	itr = 1000, burn_in = Int(itr*0.2)
)	
	states = Dict()
	function logJoint(params)	
		logPrior= logpdf(priors, params)
		logLikelihood = 0.0
		if !(model isa Nothing)
			logLikelihood = sum(logpdf.(model(params), output))
		end
		return logPrior + logLikelihood
	end

	prev_params = proposals()
	logJoint_prevs = logJoint(prev_params)
	states["itr_1"] = prev_params
	for i in 2:itr	
		current_params = proposals()	
		logJoint_curs = logJoint(current_params)
		logα = logJoint_curs - logJoint_prevs
		if (-Random.randexp() < logα)						
			prev_params = current_params
			logJoint_prevs = logJoint_curs
		end
		states["itr_$i"] = prev_params
	end
	
	return data_formatting(states, burn_in, itr)
end

function mh(priors, proposals::Distribution;
	model = nothing, 
	output = nothing,
	itr = 1000, burn_in = Int(itr*0.2)
)
	proposalf = ()->(return rand(proposals))
	return mh(priors, proposalf,
		model = model, 
		output = output,
		itr = itr, burn_in = burn_in
	)	

end

function data_formatting(states, burn_in, itr)
	chain =DataFrame();
	if(!isempty(states))
		lps = length(states["itr_1"])
		param_names =[]
		for ln in 1:lps
			push!(param_names,"param[$ln]")
		end
		chain.var = param_names
		for i in (burn_in+1):itr
			chain[!,Symbol((i-burn_in))] = rand(lps);
			jk = 0
			for ln in 1:lps			
				jk += 1
				chain[jk,(i-burn_in)+1]= states["itr_$i"][ln]
			end
		end
	end
	return chain
end
end # module
