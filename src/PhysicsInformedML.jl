module PhysicsInformedML

import SVR
import NMFk
import NTFk
import Mads
import Suppressor

function sensitivity(Xo::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray, attributes::AbstractVector; control::String="d", kw...)
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
	@Suppressor.suppress vcountt, vcountp, or2t, or2p = piml(Xo, Xin, Xsn, Xdn, times, keepcases; control=control, kw...)
	for i = 1:sz
		mask[i] = false
		local vr2t
		local vr2p
		@Suppressor.suppress vcountt, vcountp, vr2t, vr2p = piml(Xo, Xin, Xsn, Xdn, times, keepcases; control=control, mask=mask, kw...)
		mask[i] = true
		ta = abs.(or2t .- vr2t)
		pa = abs.(or2p .- vr2p)
		te = sum(ta)
		pe = sum(pa)
		@info "$(attributes[i]): $te : $pe"
		# display([ta pa])
	end
end

function piml(Xo::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray; control::String="d", plot::Bool=false, trainingrange::AbstractVector=[0., 0.05, 0.1, 0.2, 0.33], epsilon::Float64=.000000001, gamma::Float64=0.1, nc::Int64=10, mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	for r in trainingrange
		Xe = copy(Xo)
		for i = 1:ntimes
			is = sortperm(Xo[:,i])
			if control == "i"
				T = Xin[is,mask]
			elseif control == "s"
				T = Xsn[is,mask]
			elseif control == "d"
				T = Xdn[i,is,mask]
			elseif control == "a"
				T = reshape(permutedims(Xdn[:,:,mask], [2,1,3]), ncases, (ntimes * size(Xdn[:,:,mask], 3)))[is,:]
			elseif control == "is"
				T = [Xin[is,:] Xsn[is,:]]
			elseif control == "id"
				T = [Xin[is,:] Xdn[i,is,:]]
			elseif control == "isd"
				T = [Xin[is,:] Xsn[is,:] Xdn[i,is,:]]
			elseif control == "ia"
				T = [Xin[is,:] reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))[is,:]]
			elseif control == "sa"
				T = [Xsn[is,:] reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))[is,:]]
			elseif control == "isa"
				T = [Xin[is,:] Xsn[is,:] reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))[is,:]]
			else
				@warn "Unknown $control! Failed!"
				return nothing
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
			for k = 1:nc
				y_pr, pm = SVR.fit_test(Xo[is,i], permutedims(T), r; scale=true, quiet=true, epsilon=epsilon, gamma=gamma, keepcases=keepcases[is])
				y_pr[y_pr .< 0] .= 0
				countt += sum(.!pm)
				countp += sum(pm)
				Xe[is,i] .+= y_pr
				r2 = SVR.r2(Xo[is,i][.!pm], y_pr[.!pm])
				r2t += r2
				if plot
					Mads.plotseries([Xo[is,i] y_pr]; xmin=1, xmax=length(y_pr), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xo[is,i][.!pm], y_pr[.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
				end
				if sum(pm) > 0
					r2 = SVR.r2(Xo[is,i][pm], y_pr[pm])
					r2p += r2
					if plot
						NMFk.plotscatter(Xo[is,i][pm], y_pr[pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
					end
				end
			end
			Xe[is,i] ./= nc
			if plot
				r2 = SVR.r2(Xo[is,i][.!pm], Xe[is,i][.!pm])
				Mads.plotseries([Xo[is,i] Xe[is,i]]; xmin=1, xmax=size(Xo[:,i], 1), logy=false, names=["Truth", "Prediction"])
				NMFk.plotscatter(Xo[is,i][.!pm], Xe[is,i][.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
				if countp > 0
					r2 = SVR.r2(Xo[is,i][pm], Xe[is,i][pm])
					NMFk.plotscatter(Xo[is,i][pm], Xe[is,i][pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
				end
			end
			println(r, " ", countt / nc, " ", countp / nc, " ", times[i], " ", r2t / nc, " ", r2p / nc, " ")
			push!(vcountt, countt / nc)
			push!(vcountp, countp / nc)
			push!(vr2t, r2t / nc)
			push!(vr2p, r2p / nc)
		end
	end
	return vcountt, vcountp, vr2t, vr2p
end

function pimlt(Xo::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray; control::String="d", plot::Bool=false, trainingrange::AbstractVector=[0., 0.05, 0.1, 0.2, 0.33], epsilon::Float64=.000000001, gamma::Float64=0.1, nc::Int64=10, mask=Colon())
	ttimes = length(times)
	ncases = size(Xin, 1)
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	vr2tt = Vector{Float64}(undef, length(times))
	vr2tp = Vector{Float64}(undef, length(times))
	for r in trainingrange
		if control == "i"
			T = [repeat(times[1:ttimes]; inner=ncases) repeat(Xin[:,mask], ttimes)]
		elseif control == "s"
			T = [repeat(times[1:ttimes]; inner=ncases) repeat(Xsn[:,mask], ttimes)]
		elseif control == "d"
			T = [repeat(times[1:ttimes]; inner=ncases) reshape(permutedims(Xdn[:,:,mask], [2,1,3]), (ttimes * ncases, size(Xdn[:,:,mask], 3)))]
		elseif control == "is"
			T = [repeat(times[1:ttimes]; inner=ncases) repeat(Xin[:,mask], ttimes) repeat(Xsn[:,mask], ttimes)]
		elseif control == "id"
			T = [repeat(times[1:ttimes]; inner=ncases) repeat(Xin[:,mask], ttimes) reshape(permutedims(Xdn[:,:,mask], [2,1,3]), (ttimes * ncases), size(Xdn[:,:,mask], 3))]
		elseif control == "isd"
			T = [repeat(times[1:ttimes]; inner=ncases) repeat(Xin[:,mask], ttimes) repeat(Xsn[:,mask], ttimes) reshape(permutedims(Xdn[:,:,mask], [2,1,3]), (ttimes * ncases), size(Xdn[:,:,mask], 3))]
		else
			@warn "Unknown $control! Failed!"
			return nothing
		end
		local countt = 0
		local countp = 0
		local r2t = 0
		local r2p = 0
		local pm
		vr2tt .= 0
		vr2tp .= 0
		vy_tr = vec(Xo[:,1:ttimes])
		for k = 1:nc
			pm = SVR.get_prediction_mask(ncases, r; keepcases=keepcases)
			lpm = repeat(pm, ttimes)
			vy_pr, lpm = SVR.fit_test(vy_tr, permutedims(T), r; scale=true, quiet=true, epsilon=epsilon, gamma=gamma, pm=repeat(pm, ttimes))
			vy_pr[vy_pr .< 0] .= 0
			countt += sum(.!pm)
			countp += sum(pm)
			r2 = SVR.r2(vy_tr[.!lpm], vy_pr[.!lpm])
			r2t += r2
			if plot
				Mads.plotseries([vy_tr vy_pr]; xmin=1, xmax=length(vy_pr), logy=false, names=["Truth", "Prediction"])
				NMFk.plotscatter(vy_tr[.!lpm], vy_pr[.!lpm]; title="Training Size: $(sum(.!lpm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if sum(pm) > 0
				r2 = SVR.r2(vy_tr[lpm], vy_pr[lpm])
				r2p += r2
				plot && NMFk.plotscatter(vy_tr[lpm], vy_pr[lpm]; title="Prediction Size: $(sum(lpm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			y_pr = reshape(vy_pr, ncases, ttimes)
			for i = 1:length(times)
				r2tt = SVR.r2(Xo[.!pm,i], y_pr[.!pm,i])
				vr2tt[i] += r2tt
				if plot
					Mads.plotseries([Xo[:,i] y_pr[:,i]]; xmin=1, xmax=size(Xo[:,i], 1), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xo[.!pm,i], y_pr[.!pm,i]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2tt; sigdigits=2))")
				end
				if sum(pm) > 0
					r2tp = SVR.r2(Xo[pm,i], y_pr[pm,i])
					vr2tp[i] += r2tp
					plot && NMFk.plotscatter(Xo[pm,i], y_pr[pm,i]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2tp; sigdigits=2))")
				end
			end
		end
		println(r, " ", countt / nc, " ", countp / nc, " ", r2t / nc, " ", r2p / nc, " ")
		for i = 1:length(times)
			println(times[i], " ", vr2tt[i] / nc, " ", vr2tp[i] / nc)
		end
		push!(vcountt, countt / nc)
		push!(vcountp, countp / nc)
		push!(vr2t, r2t / nc)
		push!(vr2p, r2p / nc)
	end
	return vcountt, vcountp, vr2t, vr2p
end

end
