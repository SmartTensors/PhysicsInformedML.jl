module PhysicsInformedML

import SVR
import NMFk
import NTFk
import Mads
import Suppressor

function sensitivity(Xo::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray, attributes::AbstractVector; kw...)
	mask = trues(size(Xdn, 3))
	local vcountt
	local vcountp
	local or2t
	local or2p
	@Suppressor.suppress vcountt, vcountp, or2t, or2p = piml(Xo, Xin, Xsn, Xdn, times, keepcases; kw...)
	for i = 1:size(Xdn, 3)
		mask[i] = false
		local vr2t
		local vr2p
		@Suppressor.suppress vcountt, vcountp, vr2t, vr2p = piml(Xo, Xin, Xsn, Xdn, times, keepcases; mask=mask, kw...)
		mask[i] = true
		ta = abs.(or2t .- vr2t)
		pa = abs.(or2p .- vr2p)
		te = sum(ta)
		pe = sum(pa)
		@info "$(attributes[i]): $te : $pe"
		display([ta pa])
	end
end

function piml(Xo::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractArray, times::AbstractVector, keepcases::BitArray; plot::Bool=false, trainingrange::AbstractVector=[0., 0.05, 0.1, 0.2, 0.33], epsilon::Float64=.000000001, gamma::Float64=0.1, nc::Int64=10, mask=Colon())
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	for r in trainingrange
		Xe = copy(Xo)
		for i = 1:length(times)
			is = sortperm(Xo[:,i])
			# T = Xin[is,:]
			# T = Xsn[is,:]
			T = Xdn[i,is,mask]
			# T = [Xin[is,:] Xsn[is,:]]
			# T = [Xin[is,:] Xsn[is,:] Xdn[i,is,:]]
			# T = reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))
			# T = [Xin[is,:] reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))]
			# T = [Xin[is,:] Xsn[is,:] reshape(permutedims(Xdn, [2,1,3]), ncases, (ntimes * size(Xdn, 3)))]
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
			println(countt / nc, " ", countp / nc, " ", times[i], " ", r2t / nc, " ", r2p / nc, " ")
			push!(vcountt, countt / nc)
			push!(vcountp, countp / nc)
			push!(vr2t, r2t / nc)
			push!(vr2p, r2p / nc)
		end
	end
	return vcountt, vcountp, vr2t, vr2p
end

end
