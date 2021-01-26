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
	input = Array{Float64}(undef,0), 
	output = Array{Float64}(undef,0),
	itr = 1000, burn_in = Int(itr*0.2)
)		
This package aims to generate samples using The Metropolis–Hastings algorithm

# Inputs
- priors			: Prior distribution, eg: Normal(0.0,1.0)
- proposals			: proposals is proposal generating function. Eg: proposals() = rand(Normal(0.0,1.0))

# Keyword Arguments
- input				: input data
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
chm = model(x,ps1, ps2) = Normal.((x.*ps1 .+ ps2), 5.0)

histogram(Array(chm[1,2:end]),  title="MH", bins = 50)
"""


function mh(priors, proposals::Function;
	model = nothing, 
	input = nothing, 
	output = nothing,
	itr = 1000, burn_in = Int(itr*0.2)
)	
	states = Dict()
	function logJoint(params)	
		logPrior= sum(map(logpdf, priors, params))
		logLikelihood = 0.0
		if !(model isa Nothing)
			logLikelihood = sum(logpdf.(model(input, params...), output))
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

function mh(priors, proposals::Tuple{Vararg{Distribution}};
	model = nothing, 
	input = nothing, 
	output = nothing,
	itr = 1000, burn_in = Int(itr*0.2)
)
	proposalf = ()->(return map(rand, proposals))
	return mh(priors, proposalf,
		model = model, 
		input = input, 
		output = output,
		itr = itr, burn_in = burn_in
	)	

end

function data_formatting(states, burn_in, itr)
	chain =DataFrame();
	if(!isempty(states))
		lps = length(states["itr_1"])
		all_count = 0
		param_names =[]
		for ln in 1:lps
			for x in 1:length(states["itr_1"][ln])
				all_count += 1
				push!(param_names,"param[$ln]_[$x]")
			end
		end
		chain.var = param_names
		for i in (burn_in+1):itr
			chain[!,Symbol((i-burn_in))] = rand(all_count);
			jk = 0
			for ln in 1:lps			
				for x in 1:length(states["itr_1"][ln])
					jk += 1
					chain[jk,(i-burn_in)+1]= states["itr_$i"][ln][x]
				end
			end
		end
	end
	return chain
end
end # module
