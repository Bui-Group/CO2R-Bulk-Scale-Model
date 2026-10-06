# run_stage5.jl — Stage 5: φ_L (NP migration) + φ_s (solid Ohm's law), FITTED kinetics
# §11 Stage 5, driven by the §9.5.5 anchor-at-V_c=0 continuation.
include("common.jl")
include("stage5_diag.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage5", "cache")
const PREFIX    = "s5_"

# ── §11 gotcha #14: load the FITTED kinetics before anything else
include(joinpath(@__DIR__, "fitted_params.jl"))
set_i0_CO2R!(FITTED_i0_CO2R)
set_lambda_MHC!(FITTED_lambda)
set_HER_prefactor!(FITTED_HER_pf)
set_alpha_c_HER!(FITTED_alpha_c)

m = build_model(5)
print_kinetics_header(5, CACHE_DIR)
@printf("mesh: N_CL = %d, N_GDL = %d, DOFs = %d (expect 1480 at 80/120)\n",
        m.cl.N, m.gdl.N, m.ly.N_TOT)

res_at_V!(F, u, V) = residual_stage5!(F, u, V, m.cl, m.gm, m.c_eq, m.ly)

# ── §9.5.5 anchor at V_c = 0  ⇒  V_cathode(RHE) = +V_RHE_to_SHE_offset
V_anchor = V_RHE_to_SHE_offset
pre = Float64[]
let V = V_anchor
    while V - 0.025 > V_RHE_grid[1] + 1e-12
        V -= 0.025; push!(pre, V)
    end
end
chain = vcat([V_anchor], pre, V_RHE_grid)
emit  = vcat(falses(1 + length(pre)), trues(length(V_RHE_grid)))
@printf("continuation chain: anchor at V_c = 0 (V_RHE = %+.5f), %d pre-ramp steps, %d output voltages (%d total)\n\n",
        V_anchor, length(pre), length(V_RHE_grid), length(chain))

# ── anchor guess constructed from physics — never from a cache (gotcha #0)
u0 = u0_stage(m.ly, m.c_eq)          # equilibrium liquid + inlet gas + φ_s = φ_L = 0
phi_s_rng = (phi_s_idx(m.ly, 1)):(phi_s_idx(m.ly, m.ly.Ng))

# §9.5.9 step 3: ∂F/∂V structure probe — Stage 5 must have exactly ONE nonzero
let h = 1e-6, Fp = zeros(m.ly.N_TOT), Fm = zeros(m.ly.N_TOT)
    res_at_V!(Fp, u0, V_anchor + h); res_at_V!(Fm, u0, V_anchor - h)
    d = (Fp .- Fm) ./ (2h)
    nz = findall(x -> abs(x) > 1e-12, d)
    @printf("∂F/∂V structure probe: %d nonzero entries (guide: exactly 1) — row %s  %s\n\n",
            length(nz), string(nz), length(nz) == 1 ? "PASS" : "FAIL")
end

diags = NamedTuple[]
cb(V, u, info, Finf) = push!(diags, stage5_diagnostics(u, V, m.cl, m.gm, m.c_eq, m.ly))

t0 = time()
Vs, Us, rows, plu = run_continuation(u0, chain, emit, res_at_V!;
        blocks = weight_blocks(m.ly), n_log = 5*m.cl.N,
        cache_dir = CACHE_DIR, prefix = PREFIX,
        phi_s_range = phi_s_rng, verbose = false, on_emit = cb)
@printf("wall time: %.1f s\n", time() - t0)
export_all("stage5", Vs, Us, m.ly, m.cl, m.gm, rows)

@printf("\n=== Stage 5 summary ===\n")
@printf("converged %d/%d output voltages, mean iters = %.2f, max ||F||inf = %.3e\n",
        length(Vs), length(V_RHE_grid), sum(r.iters for r in rows)/length(rows),
        maximum(r.Finf for r in rows))

println("\n-- §11 Stage 5 checks 1–6 / §9.5.8 diagnostics --")
@printf("%9s %9s %9s %10s %10s %10s %10s %11s %11s\n",
        "V_RHE","i_CO","i_H2","dphi_L/mV","phi_L[1]","drop/mV","GDLshare","|di|/i","ionic_rel")
for d in diags
    V = d.V
    (abs(V - round(V/0.2)*0.2) < 0.013) || continue
    @printf("%9.3f %9.2f %9.2f %10.3f %10.4f %10.2f %10.2f %11.2e %11.2e\n",
            V, d.iCO/10, d.iH2/10, 1e3*d.dphi_L, d.phi_L1, 1e3*d.drop_total,
            d.gdl_share, d.di_rel, d.di_ionic_rel)
end

println("\n-- aggregate assertions --")
let
    hi = [d for d in diags if (d.iCO + d.iH2)/10 > 500]      # i_total > 500 mA/cm²
    @printf("max |Δi|/i (solid charge balance, need ≤ 1e-6)          = %.3e  %s\n",
            maximum(d.di_rel for d in diags),
            maximum(d.di_rel for d in diags) <= 1e-6 ? "PASS" : "FAIL")
    @printf("max ionic-boundary rel err (need < 0.01)                = %.3e  %s\n",
            maximum(d.di_ionic_rel for d in diags),
            maximum(d.di_ionic_rel for d in diags) < 0.01 ? "PASS" : "FAIL")
    if !isempty(hi)
        lo_, hi_ = extrema(1e3*abs(d.dphi_L) for d in hi)
        @printf("|Δφ_L| at i_total > 500 mA/cm²: %.2f – %.2f mV (need 5–25) %s\n",
                lo_, hi_, (lo_ >= 5 && hi_ <= 25) ? "PASS" : "CHECK")
    end
    @printf("Δφ_L sign (converged should be negative: φ_L[1] < 0)     : %s\n",
            all(d.dphi_L < 0 for d in diags if (d.iCO+d.iH2)/10 > 100) ? "negative PASS" : "CHECK")
    @printf("solid ohmic drop range                                  = %.1f – %.1f mV (guide 5–80)\n",
            1e3*minimum(d.drop_total for d in diags), 1e3*maximum(d.drop_total for d in diags))
    @printf("GDL share of the solid drop                             = %.2f – %.2f (guide 0.80–0.90)\n",
            minimum(d.gdl_share for d in diags), maximum(d.gdl_share for d in diags))
    @printf("c_Cs range in the reaction zone                         = %.0f – %.0f mol/m³ (guide 2000–6000)\n",
            minimum(d.cCs1 for d in diags), maximum(d.cCs_max for d in diags))

    iCOv = [d.iCO for d in diags]; iH2v = [d.iH2 for d in diags]
    FE = 100 .* iCOv ./ (iCOv .+ iH2v); kp = argmax(FE)
    dCO = diff(iCOv); dH2 = diff(iH2v)
    scCO = count(k -> sign(dCO[k]) != sign(dCO[k+1]), 1:length(dCO)-1)
    scH2 = count(k -> sign(dH2[k]) != sign(dH2[k+1]), 1:length(dH2)-1)
    @printf("sign changes di_CO/dV = %d (need 1), di_H2/dV = %d (need 0)   %s\n",
            scCO, scH2, (scCO == 1 && scH2 == 0) ? "PASS" : "FAIL")
    @printf("FE_CO peak = %.1f%% at V_RHE = %.3f\n", FE[kp], diags[kp].V)
    @printf("φ_L sign flips between neighbours                        = %d (need 0)\n",
            count(k -> sign(diags[k].phi_L1) != sign(diags[k+1].phi_L1) &&
                       abs(diags[k].phi_L1) > 1e-4, 1:length(diags)-1))
end

println("\n-- §9.5.8 AD validation: FD spot check at a converged mild-voltage state --")
let
    k = argmin(abs.(Vs .- (-0.300))); u = copy(Us[k]); V = Vs[k]
    N = m.ly.N_TOT
    Vr = Ref(V); rc!(F_, u_) = res_at_V!(F_, u_, Vr[])
    Jd = zeros(N, N)
    cfg = ForwardDiff.JacobianConfig(rc!, zeros(N), u, ForwardDiff.Chunk{12}())
    ForwardDiff.jacobian!(Jd, rc!, zeros(N), u, cfg)
    worst = 0.0
    for j in (3, 401, 900, phi_s_idx(m.ly, 5), phi_s_idx(m.ly, m.ly.N_GDL+5), phi_L_idx(m.ly, 5))
        h = 1e-7*max(abs(u[j]), 1.0)
        up = copy(u); up[j] += h; Fp = zeros(N); res_at_V!(Fp, up, V)
        um = copy(u); um[j] -= h; Fm = zeros(N); res_at_V!(Fm, um, V)
        col = (Fp .- Fm) ./ (2h); sc = maximum(abs, col)
        sc == 0 && continue
        rel = maximum(abs, col .- Jd[:, j]) / sc
        worst = max(worst, rel)
        @printf("  col %5d: max|FD − AD|/scale = %.3e\n", j, rel)
    end
    @printf("  -> %s\n", worst < 1e-5 ? "PASS" : "FAIL")
end

println("\n-- Stage 4 → Stage 5 current change (§11 Stage 5 check 5) --")
for V in (-1.205, -1.644, -2.079, -2.425)
    local k = argmin(abs.(Vs .- V))
    u5 = Us[k]
    u4 = load_solution(joinpath(@__DIR__, "output_fit", "cache"), "fit_", Vs[k])
    a = compute_currents5(u5, m.ly, m.cl)
    b = compute_currents(u4, Vs[k] - V_RHE_to_SHE_offset, m.cl)
    d5 = stage5_diagnostics(u5, Vs[k], m.cl, m.gm, m.c_eq, m.ly)
    c4 = unpack_log_conc(u4, m.cl.N)
    @printf("V_RHE=%.3f  S4fit: i_CO=%7.2f i_H2=%7.2f | S5: i_CO=%7.2f i_H2=%7.2f  (×%.2f, ×%.2f)\n",
            Vs[k], b[1]/10, b[2]/10, a[1]/10, a[2]/10, a[1]/b[1], a[2]/b[2])
    @printf("            c_CO3[1] %.0f → %.0f, c_Cs[1] %.0f → %.0f, H_eff[1] %.2f → %.2f, c_CO2[1] %.2f → %.2f\n",
            c4[4,1], d5.cCO3_1,
            c4[2,1]+c4[3,1]+2*c4[4,1]-c4[1,1]-C_fix*eps_I, d5.cCs1,
            sechenov_H_CO2(@view c4[:,1]), d5.H_eff_1, c4[5,1], d5.cCO2_1)
end
println("\nSTAGE 5 COMPLETE.")
