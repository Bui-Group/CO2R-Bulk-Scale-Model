# run_stage2.jl — Stage 2 driver: + Sherwood-Reynolds Neumann electrolyte BC
# §11 Stage 2 / §7.2. Same DOFs, same solver as Stage 1; only the right-face
# flux changes.

include("params.jl")
include("mesh.jl")
include("chemistry.jl")
include("kinetics.jl")
include("transport.jl")
include("assembly.jl")
include("newton_affine.jl")
include("driver.jl")

using Printf

const STAGE2_CACHE = joinpath(@__DIR__, "output_stage2", "cache")
mkpath(STAGE2_CACHE)

function save_stage2(u::Vector{Float64}, V::Float64)
    fn = joinpath(STAGE2_CACHE, @sprintf("s2_V%.6f.bin", V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
end

function main()
    @printf("=== Stage 2 ===\n")
    @printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf = %.3e, alpha_c = %.4f\n", HER_prefactor(), alpha_c_HER())
    @printf("Cache dir: %s\n", STAGE2_CACHE)

    cl = make_cl_mesh()
    N = cl.N
    @printf("N_CL = %d\n", N)

    # Biot number physicality check (§11 Stage 2), computed before solving
    Bi = [k_MT[i] * L_CL / D_eff[i] for i in 1:5]
    @printf("Biot numbers k_MT*L_CL/D_eff: %s (expect O(0.1-10))\n", round.(Bi, digits=3))

    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    u0 = make_u0_liquid_eq(c_eq_vec, N)

    res_at_V! = (F_, u_, V_) -> residual_stage2!(F_, u_, V_, cl, c_eq_vec)
    blocks = [1:5*N]

    result = run_stage_sweep!(res_at_V!, V_RHE_grid, u0; blocks=blocks, n_log_dofs=5*N,
                               save_fn=save_stage2, stage_name="Stage2")

    n_ok = count(result.converged)
    @printf("\n=== Stage 2 summary ===\n")
    @printf("Converged: %d / %d voltages\n", n_ok, length(result.voltages))
    @printf("max ||F||_inf = %.3e, mean iters/step = %.2f\n",
            maximum(result.resid), sum(result.iters)/length(result.iters))

    # Right-boundary flux physicality check at the most cathodic voltage
    u_last = result.solutions[end]
    c_last = unpack_log_conc(u_last, N)
    @printf("\nAt V_RHE=%.3f (most cathodic):\n", result.voltages[end])
    for i in 1:5
        flux_check = k_MT[i] * (c_last[i,N] - c_eq_vec[i])
        @printf("  species %s: c(L_CL)=%.4g, c_eq=%.4g, flux=k_MT*(c-c_eq)=%.4g\n",
                species_names[i], c_last[i,N], c_eq_vec[i], flux_check)
    end
    @printf("  c_OH(L_CL) should sit ABOVE c_eq_OH=%.4g (OH produced in CL): %.4g %s\n",
            c_eq_vec[2], c_last[2,N], c_last[2,N] > c_eq_vec[2] ? "PASS" : "FAIL")

    outdir = joinpath(@__DIR__, "output_stage2")
    open(joinpath(outdir, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mA_cm2,i_H2_mA_cm2,i_total_mA_cm2,FE_CO,FE_H2,converged,iters,resid")
        for (k, V) in enumerate(result.voltages)
            u = result.solutions[k]
            V_SHE = V - V_RHE_to_SHE_offset
            iCO = compute_i_CO(u, V_SHE, cl) / 10.0
            iH2 = compute_i_H2(u, V_SHE, cl) / 10.0
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
            V_SHE = V - V_RHE_to_SHE_offset
            c = unpack_log_conc(u, N)
            for ix in 1:N
                c_ix = c[:,ix]
                cCs = cCs_cell(c_ix)
                pH = -log10(max(c_ix[1]/1000.0, 1e-30))
                r_co2r = R_CO2R(c_ix[5], V_SHE, 0.0, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
                r_her = R_HER(V_SHE, 0.0, c_ix[1], a_v_CL, n_e)
                x_um = cl.x_centers[ix]*1e6
                println(f, "$V,$x_um,$pH,$(c_ix[1]),$(c_ix[2]),$(c_ix[3]),$(c_ix[4]),$cCs,$(c_ix[5]),$r_co2r,$r_her")
            end
        end
    end
    @printf("Wrote polarization.csv and profiles.csv\n")
    return result, cl
end

result2, cl2_mesh = main()
