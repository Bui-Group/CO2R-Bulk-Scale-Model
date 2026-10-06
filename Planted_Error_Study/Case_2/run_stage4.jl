# ============================================================================
# run_stage4.jl — Stage 4 driver (§11): finalize the Darcy pressure equation and
# activate gas convection. Same 1200 DOFs as Stage 3. Self-anchoring continuation
# (§9.5.5): anchor at −0.300 V_RHE from equilibrium liquid + inlet gas at uniform
# p_total, tangent predictor down the grid. The Stage 3 cache is only checked as a
# driver-order guard (and read for the currents comparison below).
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
include(joinpath(HERE, "cache.jl"))

const STAGE3_CACHE = joinpath(HERE, "output_stage3", "cache")
const STAGE4_DIR   = joinpath(HERE, "output_stage4")
const STAGE4_CACHE = joinpath(STAGE4_DIR, "cache")

# §9.5.5 anchor guess: equilibrium liquid + inlet gas at uniform p_total.
function build_u0_stage4(ly, c_eq_vec)
    u0 = zeros(ndof_stage3(ly))
    ln = log.(c_eq_vec)
    for ix in 1:ly.N_CL; u0[liquid_idx(ix,1):liquid_idx(ix,5)] .= ln; end
    for cc in 1:(ly.N_GDL + ly.N_CL)
        b = _gas_base(ly, cc)
        u0[b+1] = omega_inlet[1]; u0[b+2] = omega_inlet[2]
        u0[b+3] = omega_inlet[3]; u0[b+4] = p_total
    end
    return u0
end

# Darcy mass-balance diagnostic: inlet ρu vs net CL mass source.
function darcy_diag(u, ly, gas_mesh, cl_mesh, V_app)
    c = unpack_log_conc(u, ly.N_CL); Smass = 0.0
    for ix in 1:ly.N_CL
        cix = @view c[:, ix]
        rco = R_CO2R(cix[5], V_app, 0.0, cix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        rh  = R_HER(V_app, 0.0, cix[1], a_v_CL, n_e)
        pco2 = pCO2_atm(u, ly, ly.N_GDL + ix)
        rpt = a_v_GL * k_GL() * (sechenov_H_CO2(cix) * pco2 - cix[5])
        Smass += (-rpt*M_gas[1] + rco*M_gas[2] + rh*M_gas[3]) * cl_mesh.dx[ix]
    end
    p1 = gas_cell_p(u, ly, 1)
    uG1 = -(kappa_GDL / mu_gas) * (p1 - p_total) / gas_mesh.dx_face_left
    wf = ntuple(j -> 0.5*(omega_inlet[j] + gas_cell_omega(u, ly, 1)[j]), 4)
    rhou1 = rho_gas(0.5*(p_total + p1), Mn_from_omega(wf)) * uG1
    pcl = [gas_cell_p(u, ly, ly.N_GDL + ix) for ix in 1:ly.N_CL]
    pgdl = [gas_cell_p(u, ly, ig) for ig in 1:ly.N_GDL]
    return rhou1, Smass, (maximum(pcl) - minimum(pcl)), (p_total - minimum(pgdl)), pcl[end]
end

function run_stage4()
    println("="^70); println("=== Stage 4 ===")
    @printf("i0_CO2R=%.0f A/m², λ_MHC=%.2f eV, HER_pf=%.3e, α_c=%.4f, k_GL=%.2e\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER(), k_GL())
    @printf("κ_GDL=%.2e m², κ_CL=%.2e m², μ_gas=%.2e Pa·s\n", kappa_GDL, kappa_CL, mu_gas)
    isdir(STAGE3_CACHE) && !isempty(readdir(STAGE3_CACHE)) || error("Run Stage 3 first")
    println("="^70)

    cl_mesh  = make_cl_mesh(N_CL, L_CL; q = 1.3)
    gdl_mesh = make_gdl_mesh(N_GDL, L_GDL; q = 1.3)
    gas_mesh = make_gas_mesh(gdl_mesh, cl_mesh)
    ly = GasLayout(N_CL, N_GDL)
    c_eq_vec = equilibrium_vec(1000.0, 1000.0)
    Ndof = ndof_stage3(ly)

    rows, cols = build_gas_pattern(ly)
    Jtemplate  = pattern_matrix(rows, cols, Ndof)
    plu        = AffineLU()
    blocks     = dof_blocks(4; N_CL_ = ly.N_CL, N_GDL_ = ly.N_GDL)
    @assert blocks[2] == gas_resid_idx(ly, 1, 1):gas_resid_idx(ly, ly.N_GDL, 4)
    @assert blocks[3] == gas_resid_idx(ly, ly.N_GDL + 1, 1):gas_resid_idx(ly, ly.N_GDL + ly.N_CL, 4)
    Vs = sort(collect(V_RHE_grid), by = v -> -v)

    V_ref = Ref(Vs[1])
    res!  = (F_, u_) -> residual_stage4!(F_, u_, V_ref[], cl_mesh, gas_mesh, c_eq_vec, ly)
    assemble! = make_ad_assembler(res!, Ndof; chunk = 12)

    pol = NamedTuple[]; prof_rows = String[]; gas_rows = String[]
    u  = build_u0_stage4(ly, c_eq_vec)
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jtemplate, n_log_dofs = 5*N_CL,
        on_converged = (V_now, uu, info) -> begin
            save_solution(STAGE4_CACHE, "s4", copy(uu), V_now)

            V_app = V_now - V_RHE_to_SHE_offset
            iCO = compute_i_CO(uu, V_app, cl_mesh)/10.0; iH2 = compute_i_H2(uu, V_app, cl_mesh)/10.0
            itot = iCO+iH2; FEco = itot>0 ? 100*iCO/itot : 0.0
            push!(pol, (V_RHE=V_now, V_SHE=V_app, iCO=iCO, iH2=iH2, itot=itot,
                        FE_CO=FEco, FE_H2=100-FEco, iter=info.iterations, Finf=info.residual))
            c_all = unpack_log_conc(uu, N_CL); rco, rh2 = rate_profiles(uu, V_app, cl_mesh)
            for ix in 1:N_CL
                cv = @view c_all[:, ix]
                push!(prof_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                      V_now, cl_mesh.x_centers[ix]*1e6, -log10(cv[1]/1000.0), cv[1], cv[2], cv[3],
                      cv[4], cv[5], c_Cs_from_EN(cv), rco[ix], rh2[ix]))
            end
            for cc in 1:(N_GDL+N_CL)
                w = gas_cell_omega(uu, ly, cc); x = omega_to_x(w); p = gas_cell_p(uu, ly, cc)
                push!(gas_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e",
                      V_now, gas_mesh.x_centers[cc]*1e6, x[1]*p/101325, x[2]*p/101325, x[3]*p/101325, p/101325))
            end
            @printf("V_RHE=%.3f: iCO=%7.2f iH2=%7.2f itot=%7.2f mA/cm²  FE_CO=%5.1f%%  iter=%d ‖F‖∞=%.1e\n",
                    V_now, iCO, iH2, itot, FEco, info.iterations, info.residual)
        end)
    nconv  = count(l -> l.converged, sweep_log)
    n_iter = sum(l.iterations for l in sweep_log)
    maxF   = maximum((l.residual for l in sweep_log if l.converged); init = 0.0)
    @printf("\nStage 4 sweep: %d/%d converged in %.1f s (%d Newton iterations, %.2f/step); max ‖F‖∞ = %.2e\n",
            nconv, length(Vs), time()-t0, n_iter, n_iter/length(Vs), maxF)

    mkpath(STAGE4_DIR)
    open(joinpath(STAGE4_DIR, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,iCO_mAcm2,iH2_mAcm2,itot_mAcm2,FE_CO,FE_H2,iters,Finf")
        for p in sort(pol, by=r->r.V_RHE)
            @printf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%d,%.3e\n",
                    p.V_RHE, p.V_SHE, p.iCO, p.iH2, p.itot, p.FE_CO, p.FE_H2, p.iter, p.Finf)
        end
    end
    open(joinpath(STAGE4_DIR, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_H,c_OH,c_HCO3,c_CO3,c_CO2,c_Cs,R_CO2R,R_HER")
        for row in prof_rows; println(f, row); end
    end
    open(joinpath(STAGE4_DIR, "gasprofiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pCO2_atm,pCO_atm,pH2_atm,ptot_atm")
        for row in gas_rows; println(f, row); end
    end

    # ── Physicality checks (§ Stage 4) ──
    println("\n--- Physicality checks ---")
    Vhi = Vs[end]; uhi = load_solution(STAGE4_CACHE, "s4", Vhi)
    rhou1, Smass, dpCL, dpGDL, pcl_end = darcy_diag(uhi, ly, gas_mesh, cl_mesh, Vhi - V_RHE_to_SHE_offset)
    @printf("1. Mass balance @ V_RHE=%.3f: ρu(inlet)=%.4e vs −∫S_mass dx=%.4e (rel err %.2e)\n",
            Vhi, rhou1, -Smass, abs(rhou1+Smass)/max(abs(Smass),1e-30))
    @printf("2. Pressure drop: across CL=%.3e Pa, across GDL=%.3e Pa (expect O(1–100) Pa)\n", dpCL, dpGDL)
    @printf("3. p at CL/electrolyte end = %.4f atm\n", pcl_end/101325)
    # 4. currents vs Stage 3
    s3pol = Dict{Float64,Float64}()
    if isfile(joinpath(HERE,"output_stage3","polarization.csv"))
        for ln in readlines(joinpath(HERE,"output_stage3","polarization.csv"))[2:end]
            a = split(ln,","); s3pol[parse(Float64,a[1])] = parse(Float64,a[5])
        end
    end
    maxrel = 0.0
    for p in pol
        haskey(s3pol, p.V_RHE) || continue
        maxrel = max(maxrel, abs(p.itot - s3pol[p.V_RHE])/max(s3pol[p.V_RHE],1e-9))
    end
    @printf("4. max |Δi_tot|/i_tot vs Stage 3 = %.3e (expect tiny — Darcy is a small correction)\n", maxrel)
    println("\nStage 4 complete. Plots: run plot_results.py 4")
    return nconv, length(Vs)
end

run_stage4()
