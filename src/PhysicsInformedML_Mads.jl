import Mads
import NMFk
import SVR
import Printf
import Suppressor
import DelimitedFiles
import Interpolations
import JLD

function mads(Xon::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, keepcases::BitArray, targets::AbstractVector, times::AbstractVector, Xtn::AbstractMatrix=Matrix(undef, 0, 0); obsmin::AbstractArray=Matrix(undef, 0, 0), obsmax::AbstractArray=Matrix(undef, 0, 0), case::AbstractString, control::AbstractString, filtertimes::AbstractVector=trues(length(times)), svrdir::AbstractString=joinpath(GIMI.dir, "svr"), madsdir::AbstractString=joinpath(GIMI.dir, "mads"), load::Bool=false, save::Bool=true, paramnames::Union{Nothing,AbstractVector}=nothing, obstarget::AbstractVector=targets[filtertimes], obstime::AbstractVector=times[filtertimes], mapping::Function=i->i, Xntn::AbstractMatrix=Matrix(undef, 0, 0), negpenalty::Bool=false, maxpenalty::Bool=false, parammin::AbstractArray=Vector{Float32}(undef, 0), parammax::AbstractArray=Vector{Float32}(undef, 0), input_dict::AbstractDict=Dict(), paramkey::Union{Nothing,AbstractVector}=nothing, thickness_ratio::Number=1, kw...)
	filesvrmodel = joinpath(svrdir, "$(case)_$(control).svrmodel")
	if load && isfile(filesvrmodel)
		svrmodel = SVR.loadmodel(filesvrmodel)
		if sizeof(Xtn) > 0
			svrdata = setdata(Xin, Xsn, Xdn, [times Xtn]; control=control, mask=mask)
		else
			svrdata = setdata(Xin, Xsn, Xdn, times; control=control, mask=mask)
		end
	else
		svrpred, svrmodel, svrdata = GIMI.model(Xon[:,filtertimes], Xin, Xsn, Xdn, log10.(times[filtertimes]), keepcases, Xtn; control=control, kernel_type=SVR.RBF)
		if save
			Mads.mkdir(svrdir)
			SVR.savemodel(svrmodel, filesvrmodel)
		end
	end
	nsvrparam = size(svrdata, 2) - 1
	@info("Number of training transients: $(length(times[filtertimes]))")
	@info("Number of SVR parameters: $nsvrparam")

	function svrpredict(x::AbstractVector)
		m = mapping(x)
		if x == m
			xi = [permutedims(log10.(obstime)); repeat(x, outer=(1,length(obstime)))]
		else
			if sizeof(Xntn) > 0
				xi = [permutedims(log10.(obstime)); permutedims(Xntn); repeat(x, outer=(1,length(obstime))); permutedims(m)]
			else
				xi = [permutedims(log10.(obstime)); repeat(x, outer=(1,length(obstime))); permutedims(m)]
			end
		end
		y = SVR.predict(svrmodel, xi)
		if negpenalty
			y[y .< 0] .= 0
		end
		if maxpenalty
			for i = 2:length(y)
				if y[i] > y[i-1]
					y[i] = y[i-1]
				end
			end
		end
		if sizeof(obsmin) > 0
			if length(obsmin) == length(y)
				ii = y .> obsmax
				y[ii] .= obsmax[ii]
				ii = y .< obsmin
				y[ii] .= obsmin[ii]
			else
				y = vec(NMFk.denormalizematrix_col!(permutedims(y), obsmin, obsmax))
			end
		end
		return y ./ thickness_ratio
	end

	v = Xin[1,:]
	m = mapping(v)
	if v == m
		v = vec(svrdata[1,2:end])
		nmadsparam = length(v)
		@info("Number of Mads parameters: $nmadsparam")
	else
		nmadsparam = size(m, 2) + length(v) + size(Xntn, 2)
		@info("Number of internal Mads parameters: $nmadsparam")
		@info("Number of external Mads parameters: $(length(v))")
	end
	@assert nmadsparam == nsvrparam

	if paramnames === nothing
		paramnames = ["p$i" for i=1:length(v)]
	else
		@assert length(paramnames) == l
	end
	Mads.mkdir(madsdir)
	md = Mads.createproblem(v, obstarget, svrpredict; problemname=joinpath(madsdir, "$(case)_$(control)"), obstime=obstime, paramname=paramnames, paramkey=paramkey, kw...)

	pdata, phead = DelimitedFiles.readdlm(joinpath(GIMI.dir, input_dict["dof"]["dir"], input_dict["dof"]["initials"]), ','; header=true)
	pn = convert.(String, pdata[:,1])
	pinit = convert.(Float32, pdata[:,2])
	pmin = convert.(Float32, pdata[:,3])
	pmax = convert.(Float32, pdata[:,4])
	popt = convert.(Bool, pdata[:,5])
	errorflag = false
	for (i, k) = enumerate(paramkey)
		j = indexin([k], pn)[1]
		if j === nothing
			@warn("Parameter name is not defined: $(paramkey[i])")
			errorflag = true
			continue
		end
		if pinit[j] < parammin[1,i] || pinit[j] > parammax[1,i]
			@warn("Parameter $(pn[i]) initial value ($(pinit[j])) is out of bounds (min: $(parammin[1,i]) max: $(parammax[1,i]))!")
			errorflag = true
		end
		if pmin[j] < parammin[1,i] || pmin[j] > parammax[1,i]
			@warn("Parameter $(pn[j]) min value ($(pmin[j])) is out of bounds (min: $(parammin[1,i]) max: $(parammax[1,i]))!")
			errorflag = true
		end
		if pmax[j] < parammin[1,i] || pmax[j] > parammax[1,i]
			@warn("Parameter $(pn[j]) max value ($(pmax[j])) is out of bounds (min: $(parammin[1,i]) max: $(parammax[1,i]))!")
			errorflag = true
		end
		if pmax[j] < pmin[j]
			@warn("Parameter $(pn[j]) has a bound error (min: $(pmin[j]) max: $(pmax[j]))!")
			errorflag = true
		end
		if popt[j] && pmin[j] == pmax[j]
			@warn("Parameter $(pn[j]) needs to be optimized but it is constrained to be a constrant!")
			popt[j] = false
			errorflag = true
		end
		md["Parameters"][k]["init"] = pinit[j]
		md["Parameters"][k]["min"] = pmin[j]
		md["Parameters"][k]["max"] = pmax[j]
		delete!(md["Parameters"][k], "dist")
	end
	if errorflag
		throw("Paremeter initialization data are inaccurate!")
		return
	end

	@info("Model parameters (original ranges):")
	Mads.showallparameters(md)

	jj = indexin(paramkey, pn)
	pinitn, _ = NMFk.normalizematrix_col(permutedims(pinit[jj]); amin=parammin, amax=parammax)
	pminn, _ = NMFk.normalizematrix_col(permutedims(pmin[jj]); amin=parammin, amax=parammax)
	pmaxn, _ = NMFk.normalizematrix_col(permutedims(pmax[jj]); amin=parammin, amax=parammax)
	for (i, k) = enumerate(Mads.getparamkeys(md))
		if k == pn[jj][i]
			md["Parameters"][k]["init"] = pinitn[1,i]
			md["Parameters"][k]["min"] = pminn[1,i]
			md["Parameters"][k]["max"] = pmaxn[1,i]
			!popt[i] && (md["Parameters"][k]["type"] = nothing)
		else
			@error("Parameter $(pn[i]) does not match existing parameter keys!")
		end
	end

	@info("Model parameters (normalized ranges):")
	Mads.showallparameters(md)
	@info("Number of calibration targets: $(Int.(sum(Mads.getobsweight(md))))")
	@info("Number of total observations: $(length(obstarget))")
	return md
end

function calibrate(aw...; random::Bool=true, reruns::Number=10, case::AbstractString="",  kw...)
	@info("Setup the MADS problem...")
	md = GIMI.mads(aw...; case=case, kw...)
	pe = GIMI.calinrate(md; case=case, kw...)
	return md, pe
end

function calibrate(md::AbstractDict; random::Bool=true, reruns::Number=10, case::AbstractString="", kw...)
	@info("History matching ...")
	pe, optresults = random ? Mads.calibraterandom(md, reruns; first_init=true) : Mads.calibrate(md)
	GIMI.calibrationresults(md, pe; case=case, kw...)
	return pe
end

function calibrationresults(md::AbstractDict, pe::AbstractDict; madsdir::AbstractString=joinpath(dir, "mads"), case::AbstractString="", f_calibrated_pi::AbstractString="", f_calibrated_parameters::AbstractString="", f_match::AbstractString="", parammin::AbstractArray=Vector{Float32}(undef, 0), parammax::AbstractArray=Vector{Float32}(undef, 0), plot::Bool=true)
	f = Mads.forward(md, pe)
	of = Mads.of(md, f)
	t = Mads.getobstime(md)
	fp = joinpath(madsdir, "$(case)")
	@info("History matching PI estimates are saved in $(f_calibrated_pi) ...")
	f_calibrated_pi = setfilename(f_calibrated_pi, madsdir, fp, "_calibrated_pi.csv")
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

function modelselection(Xo::AbstractMatrix, times::AbstractVector, pi_times::AbstractVector, pi_targets::AbstractVector, case::AbstractString, topcase::Integer=20; thickness_ratio::Number=1, madsdir::AbstractString=joinpath(GIMI.dir, "mads"), filename::AbstractString="", plot::Bool=true)
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