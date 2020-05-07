module PIML

import SVR
import NMFk
import NTFk
import Mads

function piml(Xo::AbstractMatrix, Xin::AbstractMatrix, Xsn::AbstractMatrix, Xdn::AbstractMatrix, times; plot::Bool=false, trainingrange=[0., 0.05, 0.1, 0.2, 0.33], epsilon = .000000001, gamma = 0.1, nc = 10)
	for r in trainingrange
		Xe = copy(Xo)
		for i = 1:length(times)
			is = sortperm(Xo[:,i])
			# T = Xin[is,:]
			# T = Xsn[is,:]
			T = Xdn[i,is,:]
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
		end
	end
end

end
