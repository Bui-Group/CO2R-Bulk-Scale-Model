# run_stage1.jl — Stage 1 driver: liquid NP diffusion-only + Dirichlet BCs
# + simplified kinetics (phi_s = V_applied_SHE, phi_L = 0). §11 Stage 1.

include("params.jl")
include("mesh.jl")
include("chemistry.jl")
include("kinetics.jl")
include("transport.jl")
include("assembly.jl")
include("newton_affine.jl")
include("driver.jl")

using Printf

const STAGE1_CACHE = joinpath(@__DIR__, "output_stage1", "cache")
mkpath(STAGE1_CACHE)

function save_stage1(u::Vector{Float64}, V::Float64)
    fn = joinpath(STAGE1_CACHE, @sprintf("s1_V%.6f.bin", V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
end

function load_stage1(V::Float64)
    fn = joinpath(STAGE1_CACHE, @sprintf("s1_V%.6f.bin", V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

function main()
    @printf("=== Stage 1 ===\n")
    @printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf = %.3e, alpha_c = %.4f\n", HER_prefactor(), alpha_c_HER())
    @printf("Cache dir: %s\n", STAGE1_CACHE)
    if isdir(STAGE1_CACHE) && !isempty(readdir(STAGE1_CACHE))
        @printf("WARNING: existing cache found — verify kinetics match before reusing.\n")
    end

    cl = make_cl_mesh()
    N = cl.N
    @printf("N_CL = %d (production mesh %s)\n", N, N == 80 ? "YES" : "NO (dev override)")

    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    Sbuf = buffer_sources(c_eq_vec)
    @printf("equilibrium check: pH=%.4f, max|S_buf|=%.3e mol/m3/s (must be < 1e-6)\n",
            eq.pH, maximum(abs.(Sbuf)))
    @assert maximum(abs.(Sbuf)) < 1e-6

    u0 = make_u0_liquid_eq(c_eq_vec, N)

    res_at_V! = (F_, u_, V_) -> residual_stage1!(F_, u_, V_, cl, c_eq_vec)

    blocks = [1:5*N]   # single DOF block: liquid log-concentrations

    result = run_stage_sweep!(res_at_V!, V_RHE_grid, u0; blocks=blocks, n_log_dofs=5*N,
                               save_fn=save_stage1, stage_name="Stage1")

    n_ok = count(result.converged)
    @printf("\n=== Stage 1 summary ===\n")
    @printf("Converged: %d / %d voltages, V_RHE in [%.3f, %.3f]\n",
            n_ok, length(result.voltages), minimum(result.voltages), maximum(result.voltages))
    @printf("max ||F||_inf over sweep = %.3e\n", maximum(result.resid))
    @printf("mean Newton iters/step = %.2f\n", sum(result.iters)/length(result.iters))

    # ── Export CSVs for plot_results.py ──────────────────────────────────
    outdir = joinpath(@__DIR__, "output_stage1")
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
    @printf("Wrote %s and %s\n", joinpath(outdir,"polarization.csv"), joinpath(outdir,"profiles.csv"))

    return result, cl
end

result1, cl1_mesh = main()
