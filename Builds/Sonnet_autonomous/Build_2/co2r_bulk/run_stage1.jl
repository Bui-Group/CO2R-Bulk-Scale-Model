# run_stage1.jl — Stage 1: liquid NP diffusion only, Dirichlet BCs, simplified kinetics. §11.

include(joinpath(@__DIR__, "params.jl"))
include(joinpath(@__DIR__, "mesh.jl"))
include(joinpath(@__DIR__, "chemistry.jl"))
include(joinpath(@__DIR__, "kinetics.jl"))
include(joinpath(@__DIR__, "transport.jl"))
include(joinpath(@__DIR__, "gas_transport.jl"))
include(joinpath(@__DIR__, "assembly.jl"))
include(joinpath(@__DIR__, "newton_affine.jl"))
include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "solver.jl"))
include(joinpath(@__DIR__, "postprocess.jl"))

using Printf

const CACHE_DIR1 = joinpath(@__DIR__, "output_stage1", "cache")

@printf("=== Stage 1 ===\n")
@printf("N_CL=%d, N_GDL=%d (production default 80/120; override with MESH_N_CL/MESH_N_GDL)\n", MESH_N_CL, MESH_N_GDL)
@printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV\n", i0_CO2R(), lambda_MHC())
@printf("HER_pf = %.3e, alpha_c = %.4f\n", HER_prefactor(), alpha_c_HER())
@printf("Cache dir: %s\n", CACHE_DIR1)
if isdir(CACHE_DIR1) && !isempty(readdir(CACHE_DIR1))
    @printf("WARNING: Existing cache found. Verify kinetics match before reusing!\n")
end

cl_mesh = make_cl_mesh(MESH_N_CL)
ly1 = Layout14(MESH_N_CL, 0, false)
c_eq_vec = equilibrium_c_vec()
@printf("c_eq = H=%.4e OH=%.4e HCO3=%.3f CO3=%.3f CO2=%.3f mol/m^3\n", c_eq_vec...)
S_check = buffer_sources(c_eq_vec)
@printf("buffer residual at equilibrium: max|S| = %.3e mol/m^3/s (must be < 1e-6)\n", maximum(abs.(S_check)))
@assert maximum(abs.(S_check)) < 1e-6

u0 = make_u0_liquid_eq(MESH_N_CL, c_eq_vec)
res_at_V! = (F, u, V) -> residual_stage1!(F, u, V, cl_mesh, c_eq_vec)

t0 = time()
Vs, sols = run_continuation!(res_at_V!, u0, V_RHE_grid[1], collect(V_RHE_grid), ly1, 5*MESH_N_CL,
                              CACHE_DIR1, "s1")
elapsed = time() - t0

@printf("\n=== Stage 1 summary ===\n")
@printf("Converged solutions: %d / %d\n", length(Vs), length(V_RHE_grid))
@printf("Voltage range: [%.3f, %.3f] V RHE\n", minimum(Vs), maximum(Vs))
@printf("Wall time: %.2f s\n", elapsed)

# ── Physicality checks (§11 Stage 1, §12) — self-verification (no human review this run) ──
println("\n--- Physicality checks ---")
i_CO_arr = Float64[]; i_H2_arr = Float64[]; i_tot_arr = Float64[]; FE_CO_arr = Float64[]
pH_range_report = Float64[]
for (k, V) in enumerate(Vs)
    u = sols[k]
    V_applied_SHE = V - V_RHE_to_SHE_offset
    phi_s_of, phi_L_of = const_phi_closures(V_applied_SHE)
    iCO = compute_i_CO(u, cl_mesh, phi_s_of, phi_L_of) / 10.0   # A/m^2 -> mA/cm^2
    iH2 = compute_i_H2(u, cl_mesh, phi_s_of, phi_L_of) / 10.0
    push!(i_CO_arr, iCO); push!(i_H2_arr, iH2); push!(i_tot_arr, iCO+iH2)
    push!(FE_CO_arr, iCO / max(iCO+iH2, 1e-12) * 100)
    c = unpack_log_conc(u, MESH_N_CL)
    pH_x = -log10.(c[1,:] ./ 1000.0)
    push!(pH_range_report, pH_x[1]); push!(pH_range_report, pH_x[end])
end
@printf("pH range across all cells/voltages: [%.2f, %.2f]  (expect roughly 9-13)\n",
        minimum(pH_range_report), maximum(pH_range_report))
@printf("Peak FE_CO = %.1f%% at V_RHE=%.3f  (expect low, ~60-70%% peak with PRE-FIT kinetics)\n",
        maximum(FE_CO_arr), Vs[argmax(FE_CO_arr)])
@printf("FE_CO near onset (first 5 converged voltages): %s\n", string(round.(FE_CO_arr[1:5], digits=1)))
@printf("i_total range: [%.4f, %.2f] mA/cm^2\n", minimum(i_tot_arr), maximum(i_tot_arr))

# Monotonicity / smoothness (sign-change count on first difference, §9.5.8-style check)
di_CO = diff(i_CO_arr)
sign_changes_CO = count(k -> sign(di_CO[k]) != sign(di_CO[k+1]) && di_CO[k] != 0 && di_CO[k+1] != 0,
                         1:length(di_CO)-1)
@printf("Sign changes in di_CO/dV: %d (Stage 1 pre-fit kinetics; not held to the Stage-5 bar)\n", sign_changes_CO)

mkpath(joinpath(@__DIR__, "output_stage1"))
open(joinpath(@__DIR__, "output_stage1", "summary.txt"), "w") do f
    println(f, "Stage 1 summary")
    println(f, "N_CL=$MESH_N_CL  converged=$(length(Vs))/$(length(V_RHE_grid))  wall=$(round(elapsed,digits=2))s")
    println(f, "i0_CO2R=$(i0_CO2R())  lambda_MHC=$(lambda_MHC())  HER_pf=$(HER_prefactor())  alpha_c=$(alpha_c_HER())")
    for (k,V) in enumerate(Vs)
        @printf(f, "%.4f  iCO=%.5f  iH2=%.5f  itot=%.5f  FE_CO=%.2f\n", V, i_CO_arr[k], i_H2_arr[k], i_tot_arr[k], FE_CO_arr[k])
    end
end
write_polarization_csv(joinpath(@__DIR__, "output_stage1", "polarization.csv"), Vs, sols, cl_mesh, stage14_phi_of)
write_liquid_profiles_csv(joinpath(@__DIR__, "output_stage1", "liquid_profiles.csv"), Vs, sols, cl_mesh, stage14_phi_of)

println("\nStage 1 driver done. Run plot_stage.py 1 to generate plots.")
