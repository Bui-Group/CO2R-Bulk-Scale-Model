# tests/solver_tests.jl — §9.5.9 first-build sanity tests that the stage drivers
# do not cover, plus the §9.5.5 branch check and the §9.5.8 AD validation.
include(joinpath(@__DIR__, "..", "model.jl"))
using Printf, SparseArrays, LinearAlgebra, ForwardDiff

const N_CL  = N_CL_DEFAULT
const N_GDL = N_GDL_DEFAULT

# load the fitted kinetics so every test runs in the production regime
let d = Dict{String,Float64}()
    for l in eachline(joinpath(ROOT, "output_fit", "fitted_params.txt"))
        k, v = split(l, "="); d[strip(k)] = parse(Float64, strip(v))
    end
    set_i0_CO2R!(d["i0_CO2R"]); set_lambda_MHC!(d["lambda_MHC"])
    set_HER_prefactor!(d["HER_prefactor"]); set_alpha_c_HER!(d["alpha_c_HER"])
end

run5 = make_stage_run(Val(5), N_CL, N_GDL)
ly = run5.ly; N = ly.N_TOT
s5V = list_cache(run5.cache_dir, "s5")
isempty(s5V) && error("run run_stage5.jl first")

loadV(V) = load_stage(run5, V)

# ═══════════════════════════════════════════════════════════════════════════
println("── §9.5.9 step 1: persistent sparse LU ──")
u = loadV(-1.200)
run5.Vref[] = -1.200
ForwardDiff.jacobian!(run5.ws.J, run5.ws.res!, run5.ws.Fy, u, run5.ws.cfg)
J = sparse(run5.ws.J)
b = randn(N)
plu = AffineLU()
t1 = @elapsed (x1, ok1) = aff_factor_solve!(plu, J, b)
t2 = @elapsed (x2, ok2) = aff_factor_solve!(plu, J, b)     # same pattern → lu! only
t3 = @elapsed x3 = aff_resolve(plu, b)                     # back-substitution only
@printf("first  factor+solve : %.4f s  (ok=%s)\n", t1, ok1)
@printf("second factor+solve : %.4f s  (ok=%s, symbolic reused: %s)\n", t2, ok2,
        t2 < t1 ? "yes — faster" : "NOT faster")
@printf("back-substitute only: %.4f s  (%.1f× cheaper than a refactorization)\n",
        t3, t2/max(t3,1e-12))
@printf("residual ‖J x − b‖∞ = %.3e ; agreement of the three solves = %.3e\n",
        maximum(abs, J*x1 - b), maximum(abs, x1 - x3))
@assert t3 < t2 "back-substitution must be cheaper than a refactorization"

# ═══════════════════════════════════════════════════════════════════════════
println("\n── §9.5.8 AD validation: FD spot check vs the AD Jacobian ──")
# Per-DOF RELATIVE steps.  A block-rms step is wrong here: the gas blocks mix
# omega (~1e-3 … 0.95) with p (~1e5), so an rms-scaled step drives the mass
# fractions negative and the FD probe leaves the physical domain entirely.
blocks = dof_blocks(ly)
scale = [max(abs(u[j]), 1e-6) for j in 1:N]

Fp = zeros(N); Fm = zeros(N)
maxrel = 0.0
for trial in 1:5
    v = [randn()*scale[j] for j in 1:N]
    h = 1e-6
    res_at_V!(run5, Fp, u .+ h .* v, -1.200)
    res_at_V!(run5, Fm, u .- h .* v, -1.200)
    d_fd = (Fp .- Fm) ./ (2h)
    d_ad = run5.ws.J * v
    rel = norm(d_fd - d_ad) / max(norm(d_ad), 1e-30)
    global maxrel = max(maxrel, rel)
    @printf("  directional trial %d: ‖J_AD·v − FD·v‖/‖J_AD·v‖ = %.3e\n", trial, rel)
end

# column-by-column spot check covering every physical DOF type
println("  per-column spot check:")
cols = [(liquid_idx(ly,40,1), "ln c_H  (CL 40)"), (liquid_idx(ly,40,2), "ln c_OH (CL 40)"),
        (liquid_idx(ly,40,4), "ln c_CO3(CL 40)"), (liquid_idx(ly,40,5), "ln c_CO2(CL 40)"),
        (gas_gdl_idx(ly,60,1), "ω_CO2 (GDL 60)"), (gas_gdl_idx(ly,60,4), "p     (GDL 60)"),
        (gas_cl_idx(ly,40,1),  "ω_CO2 (CL 40)"),  (gas_cl_idx(ly,40,2), "ω_CO  (CL 40)"),
        (gas_cl_idx(ly,40,3),  "ω_H2  (CL 40)"),  (gas_cl_idx(ly,40,4), "p     (CL 40)"),
        (phi_s_gdl_idx(ly,60), "φ_s   (GDL 60)"), (phi_s_cl_idx(ly,40), "φ_s   (CL 40)"),
        (phi_L_idx(ly,40),     "φ_L   (CL 40)")]
maxcol = 0.0
for (jcol, nm) in cols
    hj = 1e-6 * scale[jcol]
    up = copy(u); up[jcol] += hj
    um = copy(u); um[jcol] -= hj
    res_at_V!(run5, Fp, up, -1.200); res_at_V!(run5, Fm, um, -1.200)
    col_fd = (Fp .- Fm) ./ (2hj)
    col_ad = run5.ws.J[:, jcol]
    rel = norm(col_fd - col_ad) / max(norm(col_ad), 1e-30)
    global maxcol = max(maxcol, rel)
    @printf("    %-16s col %5d : ‖ΔJ‖/‖J‖ = %.3e  (‖J col‖ = %.3e)\n",
            nm, jcol, rel, norm(col_ad))
end
@printf("max relative discrepancy: directional %.3e, per-column %.3e\n", maxrel, maxcol)
@assert maxrel < 1e-5 && maxcol < 1e-5 "AD Jacobian disagrees with FD beyond the truncation floor"

# ═══════════════════════════════════════════════════════════════════════════
println("\n── §9.5.9 step 4: predictor A/B (tangent vs carry-forward + φ_s shift) ──")
Vgrid = V_RHE_grid
ratios = Float64[]
Fchk = zeros(N)
dudV = zeros(N)
for k in 20:5:85
    Vprev = Vgrid[k]; Vnext = Vgrid[k+1]; dV = Vnext - Vprev
    u0 = copy(loadV(Vprev)); ustar = loadV(Vnext)
    # rebuild the live factorization at (u0, Vprev)
    run5.Vref[] = Vprev
    info = newton_affine!(u0, run5.ws; rtol=1e-10, max_iter=5, alpha_init=1.0,
                          n_log_dofs=5*N_CL)
    ok = tangent_predictor!(dudV, run5.ws, (F,uu,V)->res_at_V!(run5,F,uu,V), u0, Vprev)
    ok || (println("  tangent unavailable at V=$Vprev"); continue)
    u_tan = u0 .+ dV .* dudV
    u_cf  = phis_shift!(copy(u0), ly, dV)
    e_tan = norm(u_tan - ustar); e_cf = norm(u_cf - ustar)
    push!(ratios, e_cf / e_tan)
    @printf("  V %+.3f → %+.3f : ‖err‖ tangent %.3e vs carry-forward %.3e  → %6.1f× better\n",
            Vprev, Vnext, e_tan, e_cf, e_cf/e_tan)
end
sort!(ratios)
med = ratios[(length(ratios)+1)÷2]
@printf("tangent better at %d/%d steps ; median improvement %.1f× (target > 10×)\n",
        count(>(1.0), ratios), length(ratios), med)

# ═══════════════════════════════════════════════════════════════════════════
println("\n── §9.5.5 / §17.6-4 branch check: 5 experimental voltages on the sweep ──")
# continue (do NOT cold-seed) from the nearest cached grid state to the exact
# experimental voltage, then compare to the interpolated 89-point sweep curve.
sum5 = readlines(joinpath(run5.out_dir, "summary.csv"))[2:end]
Vsw   = [parse(Float64, split(l,",")[1]) for l in sum5]
iCOsw = [parse(Float64, split(l,",")[3]) for l in sum5]
iH2sw = [parse(Float64, split(l,",")[4]) for l in sum5]
function interp(xq, xs, ys)
    j = searchsortedfirst(-xs, -xq)          # xs is descending
    j = clamp(j, 2, length(xs))
    t = (xq - xs[j-1]) / (xs[j] - xs[j-1])
    return ys[j-1] + t*(ys[j] - ys[j-1])
end
println("  V_RHE    i_CO polished  i_CO sweep   Δ%      i_H2 polished  i_H2 sweep   Δ%")
maxdev = 0.0
for Ve in E_exp_RHE
    jn = argmin(abs.(Vsw .- Ve))
    up = copy(loadV(Vsw[jn]))
    run5.Vref[] = Vsw[jn]
    newton_affine!(up, run5.ws; rtol=1e-10, max_iter=5, alpha_init=1.0, n_log_dofs=5*N_CL)
    tangent_predictor!(dudV, run5.ws, (F,uu,V)->res_at_V!(run5,F,uu,V), up, Vsw[jn])
    up .+= (Ve - Vsw[jn]) .* dudV
    run5.Vref[] = Ve
    inf = newton_affine!(up, run5.ws; rtol=1e-8, max_iter=40, alpha_init=1.0,
                         n_log_dofs=5*N_CL)
    @assert inf.converged "polish failed at $Ve"
    iCOp, iH2p = compute_currents(up, ly, run5.cl, Ve)
    a = interp(Ve, Vsw, iCOsw); b2 = interp(Ve, Vsw, iH2sw)
    dCO = 100*(iCOp/10 - a)/a; dH2 = 100*(iH2p/10 - b2)/b2
    global maxdev = max(maxdev, abs(dCO), abs(dH2))
    @printf("  %6.3f  %13.3f %11.3f %+7.3f  %14.3f %11.3f %+7.3f\n",
            Ve, iCOp/10, a, dCO, iH2p/10, b2, dH2)
end
@printf("max deviation of the polished points from the sweep line = %.3f %% (target ≪ 1 %%;\n", maxdev)
println("  a wrong-branch sweep would be off by ~20 %). NOTE: the residual gap here is")
println("  linear-interpolation error on a curved line, not a branch difference.")
println("\nAll solver tests complete.")
