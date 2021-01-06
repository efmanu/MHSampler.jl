module MHSampler

using Distributions

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
function mh(propPDF, priorPDF; η = 0.65, itr = 1000)
	states = Array{Float64}(undef,0)
	burn_in = Int(itr*0.2)
	current = 0.1
	for i in 2:itr
		append!(states, current)
		movement = rand(propPDF)
		
		acceptance = min(logpdf(priorPDF,movement)/logpdf(priorPDF,current),1)
		
		if random_coin(acceptance)
            current = movement
        end
	end
	return states[burn_in:itr-1]
end

function random_coin(p)
    unif = rand(Uniform(0,1))
    if unif >= p
        return false
    else
        return true
    end
end
end # module
