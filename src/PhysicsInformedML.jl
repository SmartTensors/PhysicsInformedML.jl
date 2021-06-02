module PhysicsInformedML

import SVR
import NMFk
import NTFk
import Mads
import Suppressor

function setdata(Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector; control::String="d", order=Colon(), mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	if control == "i"
		T = [repeat(times[1:ntimes]; inner=ncases) repeat(Xin[order,mask], ntimes)]
	elseif control == "s"
		T = [repeat(times[1:ntimes]; inner=ncases) repeat(Xsn[order,mask], ntimes)]
	elseif control == "d"
		T = [repeat(times[1:ntimes]; inner=ncases) reshape(permutedims(Xdn[:,order,mask], [2,1,3]), (ntimes * ncases, size(Xdn[:,order,mask], 3)))]
	elseif control == "is"
		T = [repeat(times[1:ntimes]; inner=ncases) repeat(Xin[order,mask], ntimes) repeat(Xsn[:,mask], ntimes)]
	elseif control == "id"
		T = [repeat(times[1:ntimes]; inner=ncases) repeat(Xin[order,mask], ntimes) reshape(permutedims(Xdn[:,order,mask], [2,1,3]), (ntimes * ncases), size(Xdn[:,order,mask], 3))]
	elseif control == "sd"
		T = [repeat(times[1:ntimes]; inner=ncases) repeat(Xsn[order,mask], ntimes) reshape(permutedims(Xdn[:,order,mask], [2,1,3]), (ntimes * ncases), size(Xdn[:,order,mask], 3))]
	elseif control == "isd"
		T = [repeat(times[1:ntimes]; inner=ncases) repeat(Xin[order,mask], ntimes) repeat(Xsn[:,mask], ntimes) reshape(permutedims(Xdn[:,order,mask], [2,1,3]), (ntimes * ncases), size(Xdn[:,:,mask], 3))]
	else
		@warn "Unknown $(control)! Failed!"
		T = nothing
	end
	return T
end

function setdata(i::Integer, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray; control::String="d", order=Colon(), mask=Colon())
	ntimes = size(Xdn, 1)
	ncases = size(Xin, 1)
	if control == "i"
		T = Xin[order,mask]
	elseif control == "s"
		T = Xsn[order,mask]
	elseif control == "d"
		T = Xdn[i,order,mask]
	elseif control == "a" # all transients
		T = reshape(permutedims(Xdn[:,:,mask], [2,1,3]), ncases, (ntimes * size(Xdn[:,:,mask], 3)))[order,:]
	elseif control == "is"
		T = [Xin[order,:] Xsn[order,:]]
	elseif control == "id"
		T = [Xin[order,:] Xdn[i,order,:]]
	elseif control == "sd"
		T = [Xsn[order,:] Xdn[i,order,:]]
	elseif control == "isd"
		T = [Xin[order,:] Xsn[order,:] Xdn[i,order,:]]
	elseif control == "ia"
		T = [Xin[order,:] reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))[order,:]]
	elseif control == "sa"
		T = [Xsn[order,:] reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))[order,:]]
	elseif control == "isa"
		T = [Xin[order,:] Xsn[order,:] reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))[order,:]]
	else
		@warn "Unknown $(control)! Failed!"
		T = nothing
	end
	return T
end

function setup_mask(ratio::Number, keepcases::BitArray, ncases, ntimes, ptimes::Union{Vector{Integer},AbstractUnitRange})
	pm = SVR.get_prediction_mask(ncases, ratio; keepcases=keepcases)
	lpm = Vector{Bool}(undef, 0)
	for i = 1:ntimes
		opm = (i in ptimes) ? pm : falses(length(pm))
		lpm = vcat(lpm, opm)
	end
	return pm, lpm
end

function model(Xon::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray; control::String="d", ptimes::Union{Vector{Integer},AbstractUnitRange}=1:length(times), plot::Bool=false, plottime::Bool=plot, epsilon::Float64=.000000001, gamma::Float64=0.1, mask=Colon())
	ntimes = size(Xdn, 1)
	ncases = size(Xin, 1)
	T = setdata(Xin, Xsn, Xdn, times; control=control, mask=mask)
	if isnothing(T)
		return
	end
	r = 0
	pm, lpm = setup_mask(r, keepcases, ncases, ntimes, ptimes)
	vy_tr = vec(Xon[:,1:ntimes])
	vy_pr, lpm = SVR.fit_test(vy_tr, permutedims(T); ratio=r, scale=true, quiet=true, epsilon=epsilon, gamma=gamma, pm=lpm)
	# m = SVR.train(vy_tr[.!lpm], permutedims(T[.!lpm,:]); scale=true, epsilon=epsilon, gamma=gamma)
	# vy_pr2 = SVR.predict(m, permutedims(T))
	# @assert vy_pr == vy_pr2
	vy_pr[vy_pr .< 0] .= 0
	r2 = SVR.r2(vy_tr[.!pm], vy_pr[.!pm])
	if plot
		Mads.plotseries([vy_tr vy_pr]; xmin=1, xmax=length(vy_pr), logy=false, names=["Truth", "Prediction"])
		NMFk.plotscatter(vy_tr[.!pm], vy_pr[.!pm]; title="Training Size: $(sum(.!pm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
	end
	if sum(pm) > 0
		r2 = SVR.r2(vy_tr[lpm], vy_pr[lpm])
		plot && NMFk.plotscatter(vy_tr[lpm], vy_pr[lpm]; title="Prediction Size: $(sum(pm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
	end
	y_pr = reshape(vy_pr, ncases, ntimes)
	if plottime
		for i = 1:ncases
			Mads.plotseries(permutedims([Xon[i:ncases:end,:]; y_pr[i:ncases:end,:]]); xmin=1, xmax=length(times), logy=false, names=["Truth", "Prediction"])
		end
	end
	return y_pr, m, T
end

function calibrate()
end

function sensitivity(Xon::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray, attributes::AbstractVector; control::String="d", kw...)
	if control == "i"
		sz = size(Xin, 2)
	elseif control == "s"
		sz = size(Xsn, 2)
	elseif control == "d"
		sz = size(Xdn, 3)
	elseif control == "a"
		sz = size(Xdn, 3)
	else
		@warn "Unknown $control! Failed!"
		return nothing
	end
	@assert sz == length(attributes)
	mask = trues(sz)
	local vcountt
	local vcountp
	local or2t
	local or2p
	@Suppressor.suppress vcountt, vcountp, or2t, or2p = analysis(Xon, Xin, Xsn, Xdn, times, keepcases; control=control, kw...)
	for i = 1:sz
		mask[i] = false
		local vr2t
		local vr2p
		@Suppressor.suppress vcountt, vcountp, vr2t, vr2p = analysis(Xon, Xin, Xsn, Xdn, times, keepcases; control=control, mask=mask, kw...)
		mask[i] = true
		ta = abs.(or2t .- vr2t)
		pa = abs.(or2p .- vr2p)
		te = sum(ta)
		pe = sum(pa)
		@info "$(attributes[i]): $te : $pe"
		# display([ta pa])
	end
end

function analysis(Xon::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray; control::String="d", ptimes::AbstractUnitRange=1:length(times), plot::Bool=false, trainingrange::AbstractVector=[0., 0.05, 0.1, 0.2, 0.33], epsilon::Float64=.000000001, gamma::Float64=0.1, nreruns::Int64=10, mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	for r in trainingrange
		Xe = copy(Xon)
		for i = 1:ntimes
			is = sortperm(Xon[:,i])
			T = setdata(i, Xin, Xsn, Xdn; control=control, mask=mask, order=is)
			if isnothing(T)
				return
			end
			Xen, _, _ = NMFk.normalize!(Xe; amin=0)
			if i > 1
				# T = [T Xon[is,1:i-1]]
				T = [T Xen[is,1:i-1]]
			end
			Xe[is,i] .= 0
			local countt = 0
			local countp = 0
			local r2t = 0
			local r2p = 0
			local pm
			tr = i in ptimes ? r : 0
			for k = 1:nreruns
				y_pr, pm = SVR.fit_test(Xon[is,i], permutedims(T); ratio=tr, scale=true, quiet=true, epsilon=epsilon, gamma=gamma, keepcases=keepcases[is])
				y_pr[y_pr .< 0] .= 0
				countt += sum(.!pm)
				countp += sum(pm)
				Xe[is,i] .+= y_pr
				r2 = SVR.r2(Xon[is,i][.!pm], y_pr[.!pm])
				r2t += r2
				if plot
					Mads.plotseries([Xon[is,i] y_pr]; xmin=1, xmax=length(y_pr), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xon[is,i][.!pm], y_pr[.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
				end
				if sum(pm) > 0
					r2 = SVR.r2(Xon[is,i][pm], y_pr[pm])
					r2p += r2
					if plot
						NMFk.plotscatter(Xon[is,i][pm], y_pr[pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
					end
				end
			end
			Xe[is,i] ./= nreruns
			r2 = SVR.r2(Xon[is,i][.!pm], Xe[is,i][.!pm])
			if plot
				Mads.plotseries([Xon[is,i] Xe[is,i]]; xmin=1, xmax=size(Xon[:,i], 1), logy=false, names=["Truth", "Prediction"])
				NMFk.plotscatter(Xon[is,i][.!pm], Xe[is,i][.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if countp > 0
				r2 = SVR.r2(Xon[is,i][pm], Xe[is,i][pm])
				plot && NMFk.plotscatter(Xon[is,i][pm], Xe[is,i][pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			println(r, " ", countt / nreruns, " ", countp / nreruns, " ", times[i], " ", r2t / nreruns, " ", r2p / nreruns, " ")
			push!(vcountt, countt / nreruns)
			push!(vcountp, countp / nreruns)
			push!(vr2t, r2t / nreruns)
			push!(vr2p, r2p / nreruns)
		end
	end
	return vcountt, vcountp, vr2t, vr2p
end

function analysis(Xon::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray; control::String="d", ptimes::AbstractUnitRange=1:length(times), plot::Bool=false, trainingrange::AbstractVector=[0., 0.05, 0.1, 0.2, 0.33], epsilon::Float64=.000000001, gamma::Float64=0.1, nreruns::Int64=10, mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	for r in trainingrange
		Xe = copy(Xon)
		for i = 1:ntimes
			is = sortperm(Xon[:,i])
			T = setdata(i, Xin, Xsn, Xdn; control=control, mask=mask, order=is)
			if isnothing(T)
				return
			end
			Xen, _, _ = NMFk.normalize!(Xe; amin=0)
			if i > 1
				# T = [T Xon[is,1:i-1]]
				T = [T Xen[is,1:i-1]]
			end
			Xe[is,i] .= 0
			local countt = 0
			local countp = 0
			local r2t = 0
			local r2p = 0
			local pm
			tr = i in ptimes ? r : 0
			for k = 1:nreruns
				y_pr, pm = SVR.fit_test(Xon[is,i], permutedims(T); ratio=tr, scale=true, quiet=true, epsilon=epsilon, gamma=gamma, keepcases=keepcases[is])
				y_pr[y_pr .< 0] .= 0
				countt += sum(.!pm)
				countp += sum(pm)
				Xe[is,i] .+= y_pr
				r2 = SVR.r2(Xon[is,i][.!pm], y_pr[.!pm])
				r2t += r2
				if plot
					Mads.plotseries([Xon[is,i] y_pr]; xmin=1, xmax=length(y_pr), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xon[is,i][.!pm], y_pr[.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
				end
				if sum(pm) > 0
					r2 = SVR.r2(Xon[is,i][pm], y_pr[pm])
					r2p += r2
					if plot
						NMFk.plotscatter(Xon[is,i][pm], y_pr[pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
					end
				end
			end
			Xe[is,i] ./= nreruns
			r2 = SVR.r2(Xon[is,i][.!pm], Xe[is,i][.!pm])
			if plot
				Mads.plotseries([Xon[is,i] Xe[is,i]]; xmin=1, xmax=size(Xon[:,i], 1), logy=false, names=["Truth", "Prediction"])
				NMFk.plotscatter(Xon[is,i][.!pm], Xe[is,i][.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if countp > 0
				r2 = SVR.r2(Xon[is,i][pm], Xe[is,i][pm])
				plot && NMFk.plotscatter(Xon[is,i][pm], Xe[is,i][pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			println(r, " ", countt / nreruns, " ", countp / nreruns, " ", times[i], " ", r2t / nreruns, " ", r2p / nreruns, " ")
			push!(vcountt, countt / nreruns)
			push!(vcountp, countp / nreruns)
			push!(vr2t, r2t / nreruns)
			push!(vr2p, r2p / nreruns)
		end
	end
	return vcountt, vcountp, vr2t, vr2p
end

function analysis_transient(Xon::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray; control::String="d", ptimes::Union{Vector{Integer},AbstractUnitRange}=1:length(times), plot::Bool=false, plottime::Bool=plot, trainingrange::AbstractVector=[0., 0.05, 0.1, 0.2, 0.33], epsilon::Float64=.000000001, gamma::Float64=0.1, nreruns::Int64=10, mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	vr2tt = Vector{Float64}(undef, length(times))
	vr2tp = Vector{Float64}(undef, length(times))
	for r in trainingrange
		T = setdata(Xin, Xsn, Xdn, times; control=control, mask=mask)
		if isnothing(T)
			return
		end
		local countt = 0
		local countp = 0
		local r2t = 0
		local r2p = 0
		local pm
		vr2tt .= 0
		vr2tp .= 0
		vy_tr = vec(Xon[:,1:ntimes])
		for k = 1:nreruns
			pm, lpm = setup_mask(r, keepcases, ncases, ntimes, ptimes)
			vy_pr, lpm = SVR.fit_test(vy_tr, permutedims(T); ratio=r, scale=true, quiet=true, epsilon=epsilon, gamma=gamma, pm=lpm)
			vy_pr[vy_pr .< 0] .= 0
			countt += sum(.!pm)
			countp += sum(pm)
			r2 = SVR.r2(vy_tr[.!lpm], vy_pr[.!lpm])
			r2t += r2
			if plot
				Mads.plotseries([vy_tr vy_pr]; xmin=1, xmax=length(vy_pr), logy=false, names=["Truth", "Prediction"])
				NMFk.plotscatter(vy_tr[.!lpm], vy_pr[.!lpm]; title="Training Size: $(sum(.!pm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if sum(pm) > 0
				r2 = SVR.r2(vy_tr[lpm], vy_pr[lpm])
				r2p += r2
				plot && NMFk.plotscatter(vy_tr[lpm], vy_pr[lpm]; title="Prediction Size: $(sum(pm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			y_pr = reshape(vy_pr, ncases, ntimes)
			for i = 1:length(times)
				opm = (i in ptimes) ? pm : falses(length(pm))
				r2tt = SVR.r2(Xon[.!opm,i], y_pr[.!opm,i])
				vr2tt[i] += r2tt
				if plottime
					Mads.plotseries([Xon[:,i] y_pr[:,i]]; xmin=1, xmax=size(Xon[:,i], 1), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xon[.!opm,i], y_pr[.!opm,i]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2tt; sigdigits=2))")
				end
				if i in ptimes
					r2tp = SVR.r2(Xon[opm,i], y_pr[opm,i])
					vr2tp[i] += r2tp
					plottime && NMFk.plotscatter(Xon[opm,i], y_pr[opm,i]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2tp; sigdigits=2))")
				end
			end
		end
		# println(r, " ", countt / nreruns, " ", countp / nreruns, " ", r2t / nreruns, " ", r2p / nreruns, " ")
		for i = 1:length(times)
			println(r, " ", countt / nreruns, " ", countp / nreruns, " ", times[i], " ", vr2tt[i] / nreruns, " ", (i in ptimes) ? vr2tp[i] / nreruns : NaN)
		end
		push!(vcountt, countt / nreruns)
		push!(vcountp, countp / nreruns)
		push!(vr2t, r2t / nreruns)
		push!(vr2p, r2p / nreruns)
	end
	return vcountt, vcountp, vr2t, vr2p
end

end