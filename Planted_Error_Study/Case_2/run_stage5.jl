# ============================================================================
# run_stage5.jl — Stage 5 driver (§11 Stage 5, §3.1, §3.4, §9.5): full coupled
# electrochemistry with FITTED kinetics. φ_L (NP migration) + φ_s (solid Ohm's
# law) as monolithic DOFs (1480 total).
# Unified continuation (§9.5.5): anchor ONCE at V_c = 0 from a guess constructed
# from physics (never from a cache — gotcha #0/#9), pre-ramp in −0.025 V steps to
# the first output voltage, then walk the 89-V grid with the tangent predictor
# (φ_s-shift fallback when no tangent is available). Post-solve residual assertion
# ‖F‖∞ ≤ 1e-6 at every emitted voltage (§9.5.6); only grid voltages are saved.
# ============================================================================

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "gas_transport.jl"))
include(joinpath(HERE, "solver_sparse.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "potentials.jl"))
include(joinpath(HERE, "cache.jl"))

const FIT_DIR      = joinpath(HERE, "output_fit")
const STAGE5_DIR   = joinpath(HERE, "output_stage5")
const STAGE5_CACHE = joinpath(STAGE5_DIR, "cache")

function load_fitted!()
    fn = joinpath(FIT_DIR, "fitted_params.txt")
    isfile(fn) || error("No fitted_params.txt — run run_fit.jl first")
    for ln in readlines(fn)
        isempty(strip(ln)) && continue
        k, v = split(ln, "="); val = parse(Float64, v)
        k == "i0_CO2R"       && set_i0_CO2R!(val)
        k == "lambda_MHC"    && set_lambda_MHC!(val)
        k == "HER_prefactor" && set_HER_prefactor!(val)
        k == "alpha_c_HER"   && set_alpha_c_HER!(val)
    end
end

# Diagnostics from a converged Stage 5 state.
function stage5_diag(u, ly, cl_mesh, gas_mesh, c_eq_vec, V_RHE)
    N = ly.N_CL; V_app = V_RHE - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N)
    iCO = compute_i_CO_s5(u, ly, cl_mesh); iH2 = compute_i_H2_s5(u, ly, cl_mesh)
    i_far = iCO + iH2                                        # A/m² (= n_e F ∫(R_CO2R+R_HER)dx)
    phiL = [u[phi_L_idx5(ly, ix)] for ix in 1:N]
    dphiL = phiL[1] - phiL[N]
    phis_cl_end = u[phi_s_cl_idx5(ly, N)]
    phis_drop = phis_cl_end - V_app                          # solid ohmic rise from collector
    # charge conservation at collector: |i_s_face[1]| vs i_far
    phis_gdl1 = u[phi_s_gdl_idx5(ly, 1)]
    i_s1 = -sigma_s_eff_GDL * (phis_gdl1 - V_app) / gas_mesh.dx_face_left
    charge_relerr = abs(abs(i_s1) - abs(i_far)) / max(abs(i_far), 1e-30)
    # ionic current at right boundary (includes Cs⁺ — see potentials.jl fix)
    iIonR = 0.0
    for i in 1:5
        L_eff = D_eff[i]/k_MT[i]
        Ji = np_flux(c[i,N], c_eq_vec[i], phiL[N], 0.0, D_eff[i], z_charge[i], L_eff)
        iIonR += F_const * z_charge[i] * Ji
    end
    c_Cs_N = c_Cs_from_EN(@view c[:,N])
    c_Cs_eq = c_eq_vec[2] + c_eq_vec[3] + 2*c_eq_vec[4] - c_eq_vec[1] - C_fix*eps_I
    iIonR += F_const * 1.0 * np_flux(c_Cs_N, c_Cs_eq, phiL[N], 0.0, D_eff_Cs, 1.0, L_eff_Cs)
    ionic_relerr = abs(abs(i_far) - abs(iIonR)) / max(abs(i_far), 1e-30)
    return (iCO=iCO/10, iH2=iH2/10, dphiL=dphiL, phis_drop=phis_drop,
            charge_relerr=charge_relerr, ionic_relerr=ionic_relerr)
end

function run_stage5()
    println("="^70); println("=== Stage 5 ===")
    load_fitted!()
    @printf("FITTED kinetics: i0=%.2f A/m², λ=%.4f eV, HER_pf=%.4e, α_c=%.5f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    @printf("Mesh: N_CL=%d, N_GDL=%d\n", N_CL, N_GDL)
    isdir(joinpath(FIT_DIR, "cache")) && !isempty(readdir(joinpath(FIT_DIR, "cache"))) ||
        error("No output_fit/cache — run run_fit.jl first (fitted kinetics + the fitted-kinetics Stage-4 reference sweep)")
    println("="^70)

    cl_mesh  = make_cl_mesh(N_CL, L_CL; q = 1.3)
    gdl_mesh = make_gdl_mesh(N_GDL, L_GDL; q = 1.3)
    gas_mesh = make_gas_mesh(gdl_mesh, cl_mesh)
    ly = GasLayout(N_CL, N_GDL)
    c_eq_vec = equilibrium_vec(1000.0, 1000.0)
    Ndof = ndof_stage5(ly)
    @printf("Stage 5 DOFs = %d\n", Ndof)
    nlog = 5*N_CL

    # §9.5.3 weight blocks; assert they match the DOF layout of potentials.jl
    blocks = dof_blocks(5; N_CL_ = ly.N_CL, N_GDL_ = ly.N_GDL)
    @assert blocks[2] == gas_resid_idx(ly, 1, 1):gas_resid_idx(ly, ly.N_GDL, 4)
    @assert blocks[3] == gas_resid_idx(ly, ly.N_GDL + 1, 1):gas_resid_idx(ly, ly.N_GDL + ly.N_CL, 4)
    @assert blocks[4] == phi_s_gdl_idx5(ly, 1):phi_s_gdl_idx5(ly, ly.N_GDL)
    @assert blocks[5] == phi_s_cl_idx5(ly, 1):phi_s_cl_idx5(ly, ly.N_CL)
    @assert blocks[6] == phi_L_idx5(ly, 1):phi_L_idx5(ly, ly.N_CL)
    @assert last(blocks[6]) == Ndof

    # ── V_RHE chain: anchor at V_c = 0 (V_RHE = offset), pre-ramp in −0.025 V steps
    #    down to the first output point, then the 89 output voltages ──
    offset   = V_RHE_to_SHE_offset
    Vs       = sort(collect(V_RHE_grid), by = v -> -v)          # mild → cathodic
    V_anchor = offset                                           # V_c = V_RHE − offset = 0
    n_pre    = floor(Int, (V_anchor - Vs[1]) / 0.025 - 1e-9)
    pre_ramp = [V_anchor - 0.025*s for s in 0:n_pre]
    chain    = vcat(pre_ramp, Vs)
    out_set  = Set(round.(Vs, digits=6))
    @printf("V_c chain: %d points (%d pre-anchor + %d output), V_c %.3f → %.3f\n",
            length(chain), length(pre_ramp), length(Vs), chain[1] - offset, chain[end] - offset)

    V_ref = Ref(V_anchor)
    res!  = (F_, u_) -> residual_stage5!(F_, u_, V_ref[], cl_mesh, gas_mesh, c_eq_vec, ly)

    # anchor guess at V_c = 0 (§9.5.5): equilibrium liquid, inlet gas, φ_s = φ_L = 0 — from physics, not a cache
    u = make_u0_stage5_anchor(ly, c_eq_vec)

    # fixed sparsity pattern by probing (§9.3.3): union of the dense-AD nonzeros at the anchor
    # voltage and at the first output voltage, each also at a perturbed state — φ_s/φ_L shifted so
    # the migration and kinetic cross-terms appear, and a small pressure ramp so the Darcy
    # convective cross-terms (∝ u_G, identically zero at uniform p) appear.
    perturb = zeros(Ndof)
    for ix in 1:N_CL; perturb[phi_L_idx5(ly, ix)] = 0.005; perturb[phi_s_cl_idx5(ly, ix)] = 0.01; end
    for ig in 1:N_GDL; perturb[phi_s_gdl_idx5(ly, ig)] = 0.01; end
    for cc in 1:(N_GDL+N_CL); perturb[gas_resid_idx(ly, cc, 4)] = -0.1 * cc; end
    V_ref[] = V_anchor; r1, c1 = probe_sparsity(res!, u; perturb = perturb)
    V_ref[] = Vs[1];    r2, c2 = probe_sparsity(res!, u; perturb = perturb)
    V_ref[] = V_anchor
    Jtemplate = pattern_matrix(vcat(r1, r2), vcat(c1, c2), Ndof)
    @printf("Sparsity pattern: %d nnz (%.2f%% dense)\n", nnz(Jtemplate), 100*nnz(Jtemplate)/Ndof^2)
    assemble! = make_ad_assembler(res!, Ndof; chunk = 12)
    plu = AffineLU()
    shift! = (uu, dV) -> shift_phis!(uu, dV, ly)                # §9.5.4 predictor fallback

    # ── Anchor + continuation (§9.5.5), diagnostics collected at the output voltages only ──
    pol = NamedTuple[]; prof_rows = String[]; gas_rows = String[]; phi_rows = String[]; diag = NamedTuple[]
    t0 = time()
    sweep_log = continuation_sweep!(u, chain, res!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jtemplate, n_log_dofs = nlog,
        shift_fallback! = shift!,
        on_converged = (V_RHE_now, uu, info) -> begin
            if round(V_RHE_now, digits=6) in out_set
                save_solution(STAGE5_CACHE, "s5", copy(uu), V_RHE_now)
                d = stage5_diag(uu, ly, cl_mesh, gas_mesh, c_eq_vec, V_RHE_now)
                itot = d.iCO + d.iH2; FEco = itot > 0 ? 100*d.iCO/itot : 0.0
                push!(pol, (V_RHE=V_RHE_now, V_SHE=V_RHE_now-offset, iCO=d.iCO, iH2=d.iH2,
                            itot=itot, FE_CO=FEco, FE_H2=100-FEco, Finf=info.residual))
                push!(diag, (V_RHE=V_RHE_now, dphiL=d.dphiL, phis_drop=d.phis_drop,
                             charge=d.charge_relerr, ionic=d.ionic_relerr))
                c_all = unpack_log_conc(uu, N_CL)
                for ix in 1:N_CL
                    cv = @view c_all[:, ix]
                    push!(prof_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                          V_RHE_now, cl_mesh.x_centers[ix]*1e6, -log10(cv[1]/1000.0), cv[1], cv[2],
                          cv[3], cv[4], cv[5], c_Cs_from_EN(cv),
                          R_CO2R(cv[5], uu[phi_s_cl_idx5(ly,ix)], uu[phi_L_idx5(ly,ix)], cv[1], a_v_CL, lambda_MHC(), i0_CO2R(), 2),
                          R_HER(uu[phi_s_cl_idx5(ly,ix)], uu[phi_L_idx5(ly,ix)], cv[1], a_v_CL, 2)))
                    push!(phi_rows, @sprintf("%.6f,%.6f,%.8e",
                          V_RHE_now, cl_mesh.x_centers[ix]*1e6, uu[phi_L_idx5(ly,ix)]))
                end
                for cc in 1:(N_GDL+N_CL)
                    w = gas_cell_omega(uu, ly, cc); x = omega_to_x(w); pp = gas_cell_p(uu, ly, cc)
                    ps = cc <= N_GDL ? uu[phi_s_gdl_idx5(ly,cc)] : uu[phi_s_cl_idx5(ly,cc-N_GDL)]
                    push!(gas_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e",
                          V_RHE_now, gas_mesh.x_centers[cc]*1e6, x[1]*pp/101325, x[2]*pp/101325,
                          x[3]*pp/101325, pp/101325, ps))
                end
                @printf("V_RHE=%.3f: iCO=%7.2f iH2=%7.2f mA/cm²  FE_CO=%5.1f%%  Δφ_L=%+.1f mV  φ_s drop=%.1f mV  chg=%.1e ion=%.1e it=%d\n",
                        V_RHE_now, d.iCO, d.iH2, FEco, d.dphiL*1000, d.phis_drop*1000,
                        d.charge_relerr, d.ionic_relerr, info.iterations)
            end
        end)
    out_log = [l for l in sweep_log if round(l.V, digits=6) in out_set]
    nconv   = count(l -> l.converged, out_log); nfail = length(out_log) - nconv
    n_iter  = sum(l.iterations for l in sweep_log)
    maxF    = maximum((l.residual for l in out_log if l.converged); init = 0.0)
    @printf("\nStage 5: %d/%d output voltages converged (%d failed) in %.1f s (%d chain points, %d Newton iterations, %.2f/step); max ‖F‖∞ = %.2e\n",
            nconv, length(Vs), nfail, time()-t0, length(chain), n_iter, n_iter/length(chain), maxF)

    # ── Export ──
    mkpath(STAGE5_DIR)
    open(joinpath(STAGE5_DIR, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,iCO_mAcm2,iH2_mAcm2,itot_mAcm2,FE_CO,FE_H2,iters,Finf")
        for p in sort(pol, by=r->r.V_RHE)
            @printf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,0,%.3e\n",
                    p.V_RHE, p.V_SHE, p.iCO, p.iH2, p.itot, p.FE_CO, p.FE_H2, p.Finf)
        end
    end
    open(joinpath(STAGE5_DIR, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_H,c_OH,c_HCO3,c_CO3,c_CO2,c_Cs,R_CO2R,R_HER")
        for row in prof_rows; println(f, row); end
    end
    open(joinpath(STAGE5_DIR, "gasprofiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pCO2_atm,pCO_atm,pH2_atm,ptot_atm,phi_s")
        for row in gas_rows; println(f, row); end
    end
    open(joinpath(STAGE5_DIR, "phiL.csv"), "w") do f
        println(f, "V_RHE,x_um,phi_L")
        for row in phi_rows; println(f, row); end
    end

    # ── Physicality checks (§11 Stage 5) ──
    println("\n--- Physicality checks ---")
    if isempty(diag)
        println("(no converged output voltages — checks skipped)")
    else
        maxchg = maximum(d->d.charge, diag); maxion = maximum(d->d.ionic, diag)
        @printf("1. Charge conservation |i_s(collector)|=|i_far|: max rel err = %.2e (target ≤1e-6)\n", maxchg)
        @printf("2. Ionic current at RHS = i_far: max rel err = %.2e (target <1e-2)\n", maxion)
        # Δφ_L at high-current voltages (i_total>500 mA/cm²)
        polD = Dict(p.V_RHE => p.itot for p in pol)
        hiV = [d for d in diag if get(polD, d.V_RHE, 0.0) > 500]
        if !isempty(hiV)
            dphis = [abs(d.dphiL)*1000 for d in hiV]
            @printf("3. |Δφ_L| at i_total>500 mA/cm²: [%.1f, %.1f] mV (target 5–25 mV)\n",
                    minimum(dphis), maximum(dphis))
        end
        sd = [abs(d.phis_drop)*1000 for d in diag]
        @printf("4. Solid ohmic drop range: [%.1f, %.1f] mV\n", minimum(sd), maximum(sd))
    end
    println("\nStage 5 complete. Plots: run plot_results.py 5")
    return nconv, length(Vs)
end

run_stage5()
