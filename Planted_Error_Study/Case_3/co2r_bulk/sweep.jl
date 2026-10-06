# sweep.jl — shared Stages 1–4 sweep driver: a thin wrapper around the §9.5.5 unified
# continuation (`continuation_sweep!`). Anchor at the mildest grid voltage (−0.300 V_RHE) from
# the stage's constructed guess (equilibrium liquid [+ inlet gas], §9.5.5 table — self-anchoring,
# no previous-stage cache needed), then tangent-predictor continuation (§9.5.4) down the grid
# with the §9.5.6 post-solve assertion and halve-dV recovery at every step. Every converged
# solution is cached and its currents reported.

"""
    run_sweep!(residual_at_V!, u0, V_grid; Jpat, plu, blocks, n_log_dofs, save_fn, current_fn,
               label) -> (results, n_conv, n_tot, maxres, sweep_log)

`residual_at_V!(F, u, V)` → in-place residual at voltage V (RHE).
`u0`              → anchor guess at the mildest grid voltage; overwritten in place with the last
                    converged state.
`Jpat`            → fixed sparsity pattern (`probe_sparsity`), filled in place by the AD assembler.
`current_fn(u,V)` → tuple of reported currents (e.g. (iCO_mAcm2, iH2_mAcm2)).
`save_fn(u,V)`    → cache the converged solution.
"""
function run_sweep!(residual_at_V!, u0::Vector{Float64}, V_grid; Jpat::SparseMatrixCSC{Float64,Int},
                    plu::AffineLU, blocks::Vector{UnitRange{Int}}, n_log_dofs::Int,
                    save_fn, current_fn, label::String="")
    Vs = sort(collect(V_grid), by = v -> -v)      # mild → cathodic
    N = length(u0)
    V_ref = Ref(Vs[1])
    res_closure!(F, uu) = residual_at_V!(F, uu, V_ref[])
    assemble! = make_ad_assembler(res_closure!, N; chunk=12)
    results = Any[]; n_conv = Ref(0); maxres = Ref(0.0)
    t0 = time()
    sweep_log = continuation_sweep!(u0, Vs, res_closure!, V_ref;
        blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jpat, n_log_dofs=n_log_dofs,
        label="V_RHE", progress_every=typemax(Int),
        on_converged = (V, uu, info) -> begin
            save_fn(uu, V)                        # cache FIRST (§10.4)
            n_conv[] += 1; maxres[] = max(maxres[], info.residual)
            cur = current_fn(uu, V)
            push!(results, (V, cur...))
            @printf("%s V_RHE=%.3f  ‖F‖∞=%.2e iter=%2d  %s\n", label, V, info.residual,
                    info.iterations, join([@sprintf("%.2f", x) for x in cur], " "))
        end)
    n_iter = sum(l.iterations for l in sweep_log)
    @printf("%s sweep: %d/%d converged in %.1fs (anchor %d iters, %d Newton iterations, %.2f/step), max‖F‖∞=%.2e\n",
            label, n_conv[], length(Vs), time()-t0, sweep_log[1].iterations, n_iter, n_iter/length(Vs), maxres[])
    sort!(results, by = r -> r[1])
    return results, n_conv[], length(Vs), maxres[], sweep_log
end
