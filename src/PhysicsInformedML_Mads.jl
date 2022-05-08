import Mads
import NMFk
import SVR
import Printf
import Suppressor
import DelimitedFiles
import Gadfly
import JLD
import Statistics

function mads(paraminit::AbstractVector, obstarget::AbstractVector, svrmodel::SVR.svmmodel; paramkey::Union{Nothing,AbstractVector}=nothing, paramnames::Union{Nothing,AbstractVector}=nothing, parammin::AbstractArray=Vector{Float32}(undef, 0), parammax::AbstractArray=Vector{Float32}(undef, 0), obsmin::Union{Float64,AbstractArray}=Matrix(undef, 0, 0), obsmax::Union{Float64,AbstractArray}=Matrix(undef, 0, 0), obstime::Union{Nothing,AbstractVector}=nothing, case::AbstractString="case", madsdir::AbstractString=joinpath(PhysicsInformedML.dir, "mads"), kw...)
	if paramnames === nothing
		paramnames = ["p$i" for i=1:length(paraminit)]
	else
		@assert length(paramnames) == length(paraminit)
	end
	function svrpredict(x::AbstractVector)
		y = SVR.predict(svrmodel, x)
		return y
	end
	Mads.mkdir(madsdir)
	paraminitn, _ = NMFk.normalize(paraminit; amin=parammin, amax=parammax)
	obstargetn, _ = NMFk.normalize(obstarget; amin=obsmin, amax=obsmax)
	md = Mads.createproblem(vec(paraminitn), vec(obstargetn), svrpredict; problemname=joinpath(madsdir, "$(case)"), obstime=obstime, paramminorig=parammin, parammaxorig=parammax, obsminorig=obsmin, obsmaxorig=obsmax, kw...)
	@info("Model parameters:")
	Mads.showallparameters(md)
	@info("Model observations:")
	Mads.showobservations(md)
	@info("Number of calibration targets: $(Int.(sum(Mads.getobsweight(md) .> 0)))")
	@info("Number of total observations: $(length(obstarget))")
	return md
end

function calibrate(aw...; random::Bool=true, reruns::Number=10, case::AbstractString="case",  kw...)
	@info("Setup the MADS problem...")
	md = PhysicsInformedML.mads(aw...; case=case, kw...)
	pe = PhysicsInformedML.calibrate(md; case=case, kw...)
	return md, pe
end

function calibrate(md::AbstractDict; random::Bool=true, reruns::Number=10, case::AbstractString="case", kw...)
	@info("History matching ...")
	pe, optresults = random ? Mads.calibraterandom(md, reruns; first_init=true) : Mads.calibrate(md)
	PhysicsInformedML.calibrationresults(md, pe; case=case, kw...)
	return pe
end

function calibrationresults(md::AbstractDict, pe::AbstractDict; madsdir::AbstractString=joinpath(dir, "mads"), case::AbstractString="case", f_calibrated_pi::AbstractString="", f_calibrated_parameters::AbstractString="", f_match::AbstractString="", parammin::AbstractArray=Vector{Float32}(undef, 0), parammax::AbstractArray=Vector{Float32}(undef, 0), plot::Bool=true)
	f = Mads.forward(md, pe)
	of = Mads.of(md, f)
	t = Mads.getobstime(md)
	fp = joinpath(madsdir, "$(case)")
	@info("History matching PI estimates are saved in $(f_calibrated_pi) ...")
	f_calibrated_pi = setfilename(f_calibrated_pi, madsdir, fp, "_calibrated_targets.csv")
	DelimitedFiles.writedlm(f_calibrated_pi, [t collect(values(f))], ',')
	pmax = Mads.getparamsmax(md)
	pmin = Mads.getparamsmin(md)
	p = permutedims(collect(values(pe)))
	if size(parammin) == size(p)
		NMFk.denormalizematrix_col!(p, parammin, parammax)
		pmin = permutedims(NMFk.denormalizematrix_col(permutedims(pmin), parammin, parammax))
		pmax = permutedims(NMFk.denormalizematrix_col(permutedims(pmax), parammin, parammax))
		p = vec(p)
		pn = deepcopy(pe)
		mdn = deepcopy(md)
		for (i, k) in enumerate(keys(pn))
			pn[k] = p[i]
			mdn["Parameters"][k]["min"] = pmin[i]
			mdn["Parameters"][k]["max"] = pmax[i]
			delete!(mdn["Parameters"][k], "dist")
		end
		Mads.showallparameters(mdn, pn)
	else
		Mads.showallparameters(md, pe)
		pmax = zeros(length(p))
		pmax = ones(length(p))
		pn = pe
	end
	f_calibrated_parameters = setfilename(f_calibrated_parameters, madsdir, fp, "_calibrated_parameters.csv")
	@info("History matching parameter estimates are saved in $(f_calibrated_parameters) ...")
	DelimitedFiles.writedlm(f_calibrated_parameters, [collect(keys(pn)) collect(values(pn)) pmin pmax pmin.!=pmax], ',')
	if plot
		f_match = setfilename(f_match, madsdir, fp, "_match.png")
		@info("History matching results are plotted in $(f_match) ...")
		title = case != "" ? "Well: $(case) OF = $(round(of; sigdigits=2))" : "OF = $(round(of; sigdigits=2))"
		Mads.plotmatches(md, pe; title=title, filename=f_match, pointsize=4Gadfly.pt, linewidth=4Gadfly.pt, xmin=0, xmax=maximum(t), xtitle="Time [days]", ytitle="PI")
	end
end

function emcee(md::AbstractDict=Dict(); parammin::AbstractArray=Vector{Float32}(undef, 0), parammax::AbstractArray=Vector{Float32}(undef, 0), madsdir::AbstractString=joinpath(dir, "mads"), case::AbstractString="", f_emcee_pi::AbstractString="", f_emcee_parameters::AbstractString="", f_emcee_parameters_mean::AbstractString="", f_emcee_parameters_jld::AbstractString="", f_emcee_scatter::AbstractString="", f_emcee_spaghetti::AbstractString="", f_emcee_best_worst::AbstractString="", f_emcee_p10_50_90::AbstractString="", ofmax::Number=Inf, nsteps=1000000, burnin=max(Int(nsteps/100), 100), thinning=max(Int(burnin/100), 10), numwalkers=thinning, load::Bool=true, save::Bool=true, plot::Bool=true, execute::Bool=true, best_worst::Integer=0)
	case = case == "" ? "case" : case
	fp = joinpath(madsdir, "$(case)")
	f_emcee_pi = setfilename(f_emcee_pi, madsdir, fp, "_emcee_pi.csv")
	f_emcee_parameters = setfilename(f_emcee_parameters, madsdir, fp, "_emcee_parameters.csv")
	f_emcee_parameters_mean = setfilename(f_emcee_parameters_mean, madsdir, fp, "_emcee_parameters_mean.csv")
	f_emcee_parameters_jld = setfilename(f_emcee_parameters_jld, madsdir, fp, "_emcee_parameters.jld")
	f_emcee_scatter = setfilename(f_emcee_scatter, madsdir, fp, "_emcee_scatter.png")
	f_emcee_spaghetti = setfilename(f_emcee_spaghetti, madsdir, fp, "_emcee_spaghetti.png")
	f_emcee_best_worst = setfilename(f_emcee_best_worst, madsdir, fp, "_emcee_best_worst.png")
	f_emcee_p10_50_90 = setfilename(f_emcee_p10_50_90, madsdir, fp, "_emcee_p10_50_90.png")
	if load && isfile(f_emcee_parameters_jld)
		@info("Load AffineInvariantMCMC results from $f_emcee_parameters_jld ...")
		chain, o = JLD.load(f_emcee_parameters_jld, "chain",  "observations")
		@info("AffineInvariantMCMC results loaded: parameters = $(size(chain, 1)); realizations = $(size(chain, 2))")
	elseif execute
		@info("AffineInvariantMCMC analysis using $(nsteps) runs is initiated ...")
		save && @info("Results will be saved in $(f_emcee_parameters_jld) ...")
		# chain, llhoods = Mads.emceesampling(md; numwalkers=100, nsteps=10000, burnin=1000, thinning=100, seed=2016, sigma=0.01)
		chain, llhoods = Mads.emceesampling(md; numwalkers=numwalkers, nsteps=nsteps, burnin=burnin, thinning=thinning, seed=2016, sigma=0.01)
		@info("Forward predictions for AffineInvariantMCMC results: parameters = $(size(chain, 1)); realizations = $(size(chain, 2)) ...")
		o = Mads.forward(md, permutedims(chain))
		save && JLD.save(f_emcee_parameters_jld, "chain", chain, "observations", o)
	else
		@warn("AffineInvariantMCMC results file is missing $(f_emcee_parameters_jld)!")
		return nothing, nothing, nothing
	end
	t = Mads.getobstime(md)
	DelimitedFiles.writedlm(f_emcee_pi, [t o], ',')
	ofs = [Mads.of(md, o[:,i]) for i=1:size(o, 2)]
	iofs = sortperm(ofs)
	paramkey = Mads.getparamkeys(md)
	ptype = Mads.getparamstype(md)
	iopt = ptype .== "opt"
	if length(paramkey[iopt]) != size(chain, 1)
		@warn("The number of explored parameters do not match: $(length(paramkey[iopt])) vs. $(size(chain, 1))!")
	end
 	@info("Errors between AffineInvariantMCMC predictions and observations (lowest 10):")
	display(ofs[iofs][1:10])
	println()
	sofs = ofs .< ofmax
	if sum(sofs) > 0
		@info("Analyze AffineInvariantMCMC predictions ($(sum(sofs)) out of $(length(ofs))) with errors less than $(ofmax) ...")
		allobs = false
	else
		@warn("There are no AffineInvariantMCMC predictions with errors less than $(ofmax)!")
		@info("All the results will be processed ...")
		sofs = trues(length(sofs))
		allobs = true
	end
	if plot
		if allobs
			@info("Plot AffineInvariantMCMC predictions ...")
		else
			@info("Plot AffineInvariantMCMC predictions with errors less than $(ofmax): $(sum(sofs)) out of $(length(ofs)) in total ...")
		end
		Mads.spaghettiplot(md, o[:, sofs]; filename=f_emcee_spaghetti, xmin=0, xmax=maximum(t), title=case, xtitle="Time [days]", ytitle="PI")
		if best_worst > 0
			@info("Plot AffineInvariantMCMC best, PI-min PI-max predictions ...")
			ibw = sortperm(vec(o[end,:]))
			Mads.spaghettiplot(md, o[:, [iofs[1]..., ibw[best_worst]..., ibw[end - best_worst]...]]; filename=f_emcee_best_worst, xmin=0, xmax=maximum(t), title=case, xtitle="Time [days]", ytitle="PI", colors=["green"; repeat(["blue"], best_worst); repeat(["orange"], best_worst)])
		end
		@info("Plot AffineInvariantMCMC P10, P50, P90 predictions ...")
		ostd = 1.28 .* Statistics.std(o; dims=2) .* ofs[iofs[1]]
		omean = Statistics.mean(o; dims=2)
		p10 = omean .- ostd
		p10[p10 .< 0] .= 0
		p90 = omean .+ ostd
		Mads.spaghettiplot(md, [omean p10 p90]; filename=f_emcee_p10_50_90, xmin=0, xmax=maximum(t), title=case, xtitle="Time [days]", ytitle="PI", colors=["green", "blue", "orange"])
	end
	chain_orig = copy(chain)
	if size(parammin, 2) == length(ptype) && size(chain, 1) == sum(iopt)
		NMFk.denormalizematrix_row!(chain, parammin[iopt], parammax[iopt])
	else
		@warn("Parameters are not renormalized! Proper min/max parameter bounds are needed!")
	end
	DelimitedFiles.writedlm(f_emcee_parameters, chain, ',')
	if sum(sofs) > 0 && length(paramkey[iopt]) == size(chain, 1)
		@info("Average/min/max parameters for the AffineInvariantMCMC predictions with errors less than $(ofmax): $(sum(sofs)) out of $(length(ofs)) in total ...:")
		ps = [paramkey[iopt] vcat([hcat(Statistics.mean(chain[i, sofs]), minimum(chain[i, sofs]), maximum(chain[i, sofs])) for i=1:size(chain, 1)]...)]
		display(ps)
		println()
		DelimitedFiles.writedlm(f_emcee_parameters_mean, ps, ',')
	end
	if plot
		@info("AffineInvariantMCMC sampling scatter plot ...")
		Mads.scatterplotsamples(md, permutedims(chain), f_emcee_scatter)
	end
	return md, chain_orig, ofs
end

function create_compute(inputkeys::AbstractVector, inputmin::AbstractVector, inputmax::AbstractVector, input_transient_data::AbstractDict, functions_output::AbstractVector, outputmin::AbstractVector, outputmax::AbstractVector, outputlogv::AbstractVector, initial_maximum_stress::Number=20947.50; debug::Bool=false)
	@assert length(outputmin) == length(outputmax)
	@assert length(outputmin) == length(outputlogv)
	@assert length(inputmin) == length(inputmax)
	@assert length(inputmin) == length(inputkeys)
	if debug
		display([inputkeys inputmin inputmax])
	end
	function compute(x::AbstractVector)
		x[x .< 0] .= 0
		x[x .> 1] .= 1
		xo = NMFk.denormalize(x, inputmin, inputmax)
		input = OrderedCollections.OrderedDict{Symbol, Float32}()
		for (i, k) in enumerate(inputkeys)
			input[k] = xo[i]
		end
		ntimes = length(input_transient_data[:produced_volume])
		m = Matrix{Float32}(undef, ntimes, length(outputmax))
		maximum_perforation_efficiency = 1.0
		local k
		for t = 1:ntimes
			td = Dict{Symbol,Float32}()
			for k in keys(input_transient_data)
				td[k] = input_transient_data[k][t]
			end
			td[:initial_reservoir_pressure] = input_transient_data[:perforation_efficiency][1]
			td[:reservoir_pressure] = td[:perforation_efficiency]
			td[:initial_maximum_stress] = initial_maximum_stress
			td[:maximum_perforation_efficiency] = maximum_perforation_efficiency
			td[:perforation_efficiency] = compute_perforation_efficiency(; input..., td...)
			# @show td[:perforation_efficiency]
			if maximum_perforation_efficiency > td[:perforation_efficiency]
				maximum_perforation_efficiency = td[:perforation_efficiency]
			end
			inter = compute_intermediate_parameters(; input..., td...)
			output = compute_outputs_steps(functions_output; input..., td..., inter...)
			v = [collect(values(inter)); collect(values(output))]
			k = [collect(keys(inter)); collect(keys(output))]
			im = isnan.(v)
			if any(im)
				if debug
					display([k v])
					display([k[im] v[im]])
				end
				v[im] .= 0
			end
			im = isinf.(v)
			if any(im)
				if debug
					display([k v])
					display([k[im] v[im]])
				end
				v[im] .= 1
			end
			m[t,:], _, _ = NMFk.normalize!(v; amin=outputmin, amax=outputmax, logv=outputlogv)
		end
		if debug
			@info("Number of computed attributes: $(length(k))")
			display([k outputmin outputmax])
			display(m)
		end
		return m
	end
	return compute
end

function set_input_transient_matrix(input_transient_data_keys::Base.KeySet, well_attributes::AbstractVector, well_data::AbstractMatrix, well_times::AbstractVector, newtimes::AbstractVector=well_times)
	input_transient_data_dof = OrderedCollections.OrderedDict{Symbol, Vector}()
	oldtimes = well_times
	ntimes = length(newtimes)
	input_transient_matrix_dof = Matrix{Float64}(undef, ntimes, length(input_transient_data_keys))
	for (i, k) in enumerate(input_transient_data_keys)
		n = titlecase(replace(String(k), "_"=>" "))
		ind = only(indexin([n], well_attributes))
		if ind === nothing
			@warn "Attribute $n is missing"
		else
			@info n
		end
		if k == :depletion
			v = vec(well_data[:,indexin(["Reservoir Pressure"], well_attributes)])
			v = v[1] .- v
		elseif k == :residual_permeability_multiplier
			v = ones(Float32, ntimes)
		elseif k == :equivalent_time
			v1 = vec(well_data[:,indexin(["Produced Volume"], well_attributes)])
			v2 = vec(well_data[:,indexin(["Production Rate"], well_attributes)])
			v = v1 ./ v2
		elseif ind === nothing && k == :perforation_efficiency
			@warn("Perforation efficiency needs to be computed! Reservoir Pressure stored as Peforation Efficiency!")
			# v = ones(Float32, ntimes) * 0.17
			# v[1] = 1.0
			v = vec(well_data[:,indexin(["Reservoir Pressure"], well_attributes)])
		else
			v = vec(well_data[:,ind])
		end
		if  newtimes != oldtimes
			v = Interpolations.ConstantInterpolation(oldtimes, v, extrapolation_bc=Interpolations.Line()).(newtimes)
		end
		input_transient_data_dof[k] = v
		input_transient_matrix_dof[:, i] .= v
	end
	# input_transient_matrix_dof_normalized, _, _ = NMFk.normalizematrix_col(input_transient_matrix_dof)
	return input_transient_data_dof, input_transient_matrix_dof
end

function modelselection(Xo::AbstractMatrix, times::AbstractVector, pi_times::AbstractVector, pi_targets::AbstractVector, case::AbstractString, topcase::Integer=20; thickness_ratio::Number=1, madsdir::AbstractString=joinpath(PhysicsInformedML.dir, "mads"), filename::AbstractString="", plot::Bool=true)
	case = case == "" ? "case" : case
	fp = joinpath(madsdir, "$(case)")
	filename = setfilename(filename, madsdir, fp, "_model_selection_match.png")
	newpi = Interpolations.LinearInterpolation(pi_times, pi_targets, extrapolation_bc=Interpolations.Line()).(times)
	@assert size(Xo[1,:]) == size(newpi)
	s = Vector{Float32}(undef, size(Xo, 1))
	for i = 1:size(Xo, 1)
		s[i] = sum((Xo[i,:] ./ thickness_ratio .- newpi) .^ 2)
	end
	if plot
		c = Mads.plotseries(permutedims(Xo[sortperm(s)[1:topcase],:]) ./ thickness_ratio; xaxis=times, code=true, key_position=:none, quiet=true)
		Mads.plotseries(pi_targets, filename; xaxis=pi_times, plotline=false, pointsize=3Gadfly.pt, gl=c, title="Well $(case): Top $(topcase) models")
	end
	return s
end

function setfilename(fn::AbstractString, dir::AbstractString, prefix::AbstractString, suffix::AbstractString)
	fn = fn == "" ? prefix * suffix : joinpath(dir, fn)
	Mads.recursivemkdir(fn; filename=true)
	return fn
end