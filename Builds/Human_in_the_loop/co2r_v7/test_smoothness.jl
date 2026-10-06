using DelimitedFiles, Printf
for (tag,f) in (("Stage 5","output_stage5/summary.csv"), ("Fit sweep","output_fit/summary.csv"))
    S = readdlm(f, ',', skipstart=1)
    V = S[:,1]; iCO = S[:,7]; iH2 = S[:,8]
    for (nm, y) in (("i_CO", iCO), ("i_H2", iH2))
        d = diff(y)
        # local-outlier metric: each first difference vs the mean of its neighbours
        worst = 0.0; wk = 0
        for k in 2:length(d)-1
            ref = 0.5*(d[k-1]+d[k+1])
            abs(ref) < 1e-12 && continue
            r = abs(d[k]-ref)/abs(ref)
            r > worst && (worst = r; wk = k)
        end
        nsign = sum(diff(sign.(d)) .!= 0)
        @printf("%-9s %-5s: derivative sign changes=%d ; worst step-vs-neighbour-mean deviation=%.2f %% at V=%+.3f\n",
                tag, nm, nsign, 100*worst, V[wk+1])
    end
end
