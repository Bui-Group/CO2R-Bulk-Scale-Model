# run_stage3.jl — Stage 3 driver: + Stefan-Maxwell gas transport
# (pressure DOF pinned to p_total; u_G=0, no Darcy yet) + R_PT phase
# transfer. §11 Stage 3.

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

const STAGE3_CACHE = joinpath(@__DIR__, "output_stage3", "cache")
mkpath(STAGE3_CACHE)

function save_stage3(u::Vector{Float64}, V::Float64)
    fn = joinpath(STAGE3_CACHE, @sprintf("s3_V%.6f.bin", V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
end

function main()
    @printf("=== Stage 3 ===\n")
    @printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf = %.3e, alpha_c = %.4f, k_GL = %.3e\n", HER_prefactor(), alpha_c_HER(), k_GL())
    @printf("Cache dir: %s\n", STAGE3_CACHE)

    cl = make_cl_mesh(); gdl = make_gdl_mesh()
    ly = GasLayout(cl.N, gdl.N)
    Ntot = n_dofs_34(ly)
    @printf("N_CL=%d, N_GDL=%d, total DOFs=%d (production=1200 at 80/120)\n", cl.N, gdl.N, Ntot)

    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    u0 = make_u0_stage34(c_eq_vec, cl, gdl, ly)

    # Trigger JIT on the (trivially converged) inlet state — §11 Stage 3 bootstrap note
    F0 = zeros(Ntot)
    res0! = (F_, u_) -> residual_stage3!(F_, u_, V_RHE_grid[1], cl, gdl, c_eq_vec, ly)
    res0!(F0, u0)
    @printf("Residual at gas-inlet/liquid-eq state (should be ~0 for gas rows, small for liquid rows due to kinetics at -0.3V): max|F|=%.3e\n", maximum(abs, F0))

    res_at_V! = (F_, u_, V_) -> residual_stage3!(F_, u_, V_, cl, gdl, c_eq_vec, ly)

    blocks = [1:5*cl.N, (5*cl.N+1):(5*cl.N+4*gdl.N), (5*cl.N+4*gdl.N+1):Ntot]

    result = run_stage_sweep!(res_at_V!, V_RHE_grid, u0; blocks=blocks, n_log_dofs=5*cl.N,
                               save_fn=save_stage3, stage_name="Stage3")

    n_ok = count(result.converged)
    @printf("\n=== Stage 3 summary ===\n")
    @printf("Converged: %d / %d voltages\n", n_ok, length(result.voltages))
    @printf("max ||F||_inf = %.3e, mean iters/step = %.2f\n",
            maximum(result.resid), sum(result.iters)/length(result.iters))

    # Physicality check: omega_CO2 decreases from inlet through the CL at high current
    u_last = result.solutions[end]
    om_gdl1 = gdl_omega(u_last, ly, 1)
    om_cl_end = cl_omega(u_last, ly, cl.N)
    @printf("At V_RHE=%.3f: omega_CO2(GDL inlet-side cell)=%.4f, omega_CO2(CL exit)=%.4f (inlet=%.4f)\n",
            result.voltages[end], om_gdl1[1], om_cl_end[1], omega_inlet[1])
    @printf("omega_CO(CL exit)=%.4f, omega_H2(CL exit)=%.4f (both should be > 0, increasing with current)\n",
            om_cl_end[2], om_cl_end[3])

    outdir = joinpath(@__DIR__, "output_stage3")
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
            c = unpack_log_conc(u, cl.N)
            for ix in 1:cl.N
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
    @printf("Wrote polarization.csv, profiles.csv, gas_profiles.csv\n")
    return result, cl, gdl, ly
end

result3, cl3_mesh, gdl3_mesh, ly3 = main()
