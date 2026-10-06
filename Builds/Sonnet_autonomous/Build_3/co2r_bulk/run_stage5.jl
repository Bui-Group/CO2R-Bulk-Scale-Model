# run_stage5.jl — Stage 5 driver: + phi_L (NP migration) + phi_s (solid
# Ohm's law), using FITTED kinetics from run_fit.jl. §11 Stage 5.
# Seeded ONCE at V_c=0 from a physics-constructed anchor guess — never from
# a cache (§11 Stage 5 gotcha #0 / #9).

include("params.jl")
include("mesh.jl")
include("chemistry.jl")
include("kinetics.jl")
include("transport.jl")
include("gas_transport.jl")
include("assembly.jl")
include("newton_affine.jl")
include("driver.jl")

using Printf

const STAGE5_CACHE = joinpath(@__DIR__, "output_stage5", "cache")
mkpath(STAGE5_CACHE)

function save_stage5(u::Vector{Float64}, V::Float64)
    fn = joinpath(STAGE5_CACHE, @sprintf("s5_V%.6f.bin", V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
end

function load_fitted_kinetics!()
    fn = joinpath(@__DIR__, "output_fit", "fitted_params.txt")
    isfile(fn) || error("output_fit/fitted_params.txt not found — run run_fit.jl first")
    for line in eachline(fn)
        k, v = split(line, "=")
        val = parse(Float64, v)
        if k == "i0_CO2R"; set_i0_CO2R!(val)
        elseif k == "lambda_MHC"; set_lambda_MHC!(val)
        elseif k == "HER_prefactor"; set_HER_prefactor!(val)
        elseif k == "alpha_c_HER"; set_alpha_c_HER!(val)
        end
    end
end

function main()
    @printf("=== Stage 5 ===\n")
    load_fitted_kinetics!()
    @printf("Loaded FITTED kinetics: i0_CO2R=%.3f A/m^2, lambda_MHC=%.4f eV, HER_pf=%.4e, alpha_c=%.5f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    @printf("(pre-fit defaults were: i0=540, lambda=1.8, HER_pf=2.409e13, alpha_c=0.0872 — verify these differ)\n")
    @printf("Cache dir: %s\n", STAGE5_CACHE)

    cl = make_cl_mesh(); gdl = make_gdl_mesh()
    ly = Stage5Layout(cl.N, gdl.N)
    @printf("N_CL=%d, N_GDL=%d, total DOFs=%d (production=1480 at 80/120)\n", cl.N, gdl.N, ly.N_TOT)

    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    u0 = make_u0_stage5_anchor(c_eq_vec, cl, gdl, ly)

    res_at_V! = (F_, u_, V_) -> residual_stage5!(F_, u_, V_, cl, gdl, c_eq_vec, ly)
    blocks = [1:5*cl.N, (5*cl.N+1):(5*cl.N+4*gdl.N), (5*cl.N+4*gdl.N+1):(5*cl.N+4*(gdl.N+cl.N)),
              (5*cl.N+4*(gdl.N+cl.N)+1):(5*cl.N+4*(gdl.N+cl.N)+gdl.N),
              (5*cl.N+4*(gdl.N+cl.N)+gdl.N+1):(5*cl.N+4*(gdl.N+cl.N)+gdl.N+cl.N),
              (5*cl.N+4*(gdl.N+cl.N)+gdl.N+cl.N+1):ly.N_TOT]
    phi_shift! = (u_, dV_) -> phi_s_shift_stage5!(u_, ly, dV_)

    # ── §9.5.9 step 3: ∂F/∂V structure probe (should be exactly 1 nonzero) ──
    let
        Ftest = zeros(ly.N_TOT)
        Fp = zeros(ly.N_TOT); Fm = zeros(ly.N_TOT)
        res_at_V!(Fp, u0, 0.0 + 1e-6 + V_RHE_to_SHE_offset)
        res_at_V!(Fm, u0, 0.0 - 1e-6 + V_RHE_to_SHE_offset)
        dFdV = (Fp .- Fm) ./ 2e-6
        nnz_dFdV = count(x -> abs(x) > 1e-8, dFdV)
        @printf("∂F/∂V structure probe at V_c=0 anchor guess: %d nonzero entries (expect exactly 1)\n", nnz_dFdV)
    end

    result = run_stage5_sweep!(res_at_V!, V_SHE_grid, u0; blocks=blocks, n_log_dofs=5*cl.N,
                                save_fn=save_stage5, phi_s_shift! = phi_shift!, stage_name="Stage5")

    n_ok = count(result.converged)
    @printf("\n=== Stage 5 summary ===\n")
    @printf("Converged: %d / %d voltages\n", n_ok, length(result.voltages))
    @printf("max ||F||_inf = %.3e, mean iters/step (excl. anchor) = %.2f\n",
            maximum(result.resid), sum(result.iters[2:end])/max(1,length(result.iters)-1))

    # ── §11 Stage 5 physicality checks ─────────────────────────────────────
    @printf("\n--- Physicality checks ---\n")
    for k in [10, 30, 50, 70, 89]
        k > length(result.voltages) && continue
        V = result.voltages[k]; u = result.solutions[k]
        phi_L = unpack_phi_L_cl(u, ly)
        phi_s_cl_v = unpack_phi_s_cl(u, ly)
        phi_s_gdl_v = unpack_phi_s_gdl(u, ly)
        V_applied_SHE = V - V_RHE_to_SHE_offset

        dphiL = phi_L[1] - phi_L[cl.N]
        iCO = compute_i_CO_stage5(u, ly, cl); iH2 = compute_i_H2_stage5(u, ly, cl)
        i_far_total = iCO + iH2
        i_s_face1 = -sigma_s_eff_GDL * (phi_s_gdl_v[1] - V_applied_SHE) / gdl.dx_face_left
        rel_diff_current = abs(abs(i_s_face1) - abs(i_far_total)) / max(abs(i_far_total), 1e-12)

        solid_drop = phi_s_cl_v[cl.N] - V_applied_SHE

        # ionic current at right boundary
        L_eff5 = D_eff[5]/k_MT[5]
        c_last = unpack_log_conc(u, cl.N)[:,cl.N]
        i_ionic_R = 0.0
        for i in 1:5
            Leff_i = D_eff[i]/k_MT[i]
            Jb = np_flux(c_last[i], c_eq_vec[i], phi_L[cl.N], 0.0, D_eff[i], z_i[i], Leff_i)
            i_ionic_R += z_i[i]*Jb
        end
        cCs_last = cCs_cell(c_last)
        i_ionic_R *= F_const
        rel_err_ionic = abs(abs(i_far_total) - abs(i_ionic_R)) / max(abs(i_far_total), 1e-12)

        @printf("V_RHE=%.3f: i_tot=%.1f mA/cm2, Δφ_L=%.2f mV, solid drop=%.2f mV, |Δi_far_vs_is|/i=%.2e, ionic-R rel_err=%.2e\n",
                V, i_far_total/10.0, dphiL*1000, solid_drop*1000, rel_diff_current, rel_err_ionic)
    end

    outdir = joinpath(@__DIR__, "output_stage5")
    open(joinpath(outdir, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mA_cm2,i_H2_mA_cm2,i_total_mA_cm2,FE_CO,FE_H2,converged,iters,resid")
        for (k, V) in enumerate(result.voltages)
            u = result.solutions[k]
            V_SHE = V - V_RHE_to_SHE_offset
            iCO = compute_i_CO_stage5(u, ly, cl) / 10.0
            iH2 = compute_i_H2_stage5(u, ly, cl) / 10.0
            itot = iCO + iH2
            feCO = itot > 0 ? 100*iCO/itot : 0.0
            feH2 = itot > 0 ? 100*iH2/itot : 0.0
            println(f, "$V,$V_SHE,$iCO,$iH2,$itot,$feCO,$feH2,$(result.converged[k]),$(result.iters[k]),$(result.resid[k])")
        end
    end
    open(joinpath(outdir, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_H,c_OH,c_HCO3,c_CO3,c_Cs,c_CO2,R_CO2R,R_HER")
        for (k, V) in enumerate(result.voltages)
            u = result.solutions[k]
            c = unpack_log_conc(u, cl.N)
            phi_L = unpack_phi_L_cl(u, ly); phi_s_cl_v = unpack_phi_s_cl(u, ly)
            for ix in 1:cl.N
                c_ix = c[:,ix]
                cCs = cCs_cell(c_ix)
                pH = -log10(max(c_ix[1]/1000.0, 1e-30))
                r_co2r = R_CO2R(c_ix[5], phi_s_cl_v[ix], phi_L[ix], c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
                r_her = R_HER(phi_s_cl_v[ix], phi_L[ix], c_ix[1], a_v_CL, n_e)
                x_um = cl.x_centers[ix]*1e6
                println(f, "$V,$x_um,$pH,$(c_ix[1]),$(c_ix[2]),$(c_ix[3]),$(c_ix[4]),$cCs,$(c_ix[5]),$r_co2r,$r_her")
            end
        end
    end
    open(joinpath(outdir, "gas_profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,p_CO2_atm,p_CO_atm,p_H2_atm,p_total_Pa")
        for (k, V) in enumerate(result.voltages)
            u = result.solutions[k]
            for ig in 1:gdl.N
                om = gdl_omega(u, ly, ig); p = gdl_p(u, ly, ig)
                Mn = Mn_from_omega(om); x = omega_to_x(om, Mn)
                x_um = gdl.x_centers[ig]*1e6
                println(f, "$V,$x_um,$(x[1]*p/101325),$(x[2]*p/101325),$(x[3]*p/101325),$p")
            end
            for ix in 1:cl.N
                om = cl_omega(u, ly, ix); p = cl_p(u, ly, ix)
                Mn = Mn_from_omega(om); x = omega_to_x(om, Mn)
                x_um = cl.x_centers[ix]*1e6
                println(f, "$V,$x_um,$(x[1]*p/101325),$(x[2]*p/101325),$(x[3]*p/101325),$p")
            end
        end
    end
    open(joinpath(outdir, "phi_profiles.csv"), "w") do f
        println(f, "V_RHE,x_m,phi_s,phi_L")
        for (k, V) in enumerate(result.voltages)
            u = result.solutions[k]
            phi_s_gdl_v = unpack_phi_s_gdl(u, ly); phi_s_cl_v = unpack_phi_s_cl(u, ly)
            phi_L = unpack_phi_L_cl(u, ly)
            for ig in 1:gdl.N
                println(f, "$V,$(gdl.x_centers[ig]),$(phi_s_gdl_v[ig]),")
            end
            for ix in 1:cl.N
                println(f, "$V,$(cl.x_centers[ix]),$(phi_s_cl_v[ix]),$(phi_L[ix])")
            end
        end
    end
    @printf("Wrote polarization.csv, profiles.csv, gas_profiles.csv, phi_profiles.csv\n")

    # ── sign-change / smoothness check (§9.5.8) ────────────────────────────
    iCOs = [compute_i_CO_stage5(result.solutions[k], ly, cl) for k in 1:length(result.voltages)]
    iH2s = [compute_i_H2_stage5(result.solutions[k], ly, cl) for k in 1:length(result.voltages)]
    dCO = diff(iCOs); dH2 = diff(iH2s)
    sign_changes_CO = count(i -> sign(dCO[i]) != sign(dCO[i+1]) && dCO[i] != 0 && dCO[i+1] != 0, 1:length(dCO)-1)
    sign_changes_H2 = count(i -> sign(dH2[i]) != sign(dH2[i+1]) && dH2[i] != 0 && dH2[i+1] != 0, 1:length(dH2)-1)
    @printf("Sign changes in di_CO/dV = %d (expect 1), di_H2/dV = %d (expect 0)\n", sign_changes_CO, sign_changes_H2)

    return result, cl, gdl, ly
end

result5, cl5_mesh, gdl5_mesh, ly5 = main()
