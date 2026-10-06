# run_stage4.jl — Stage 4: Darcy pressure finalized (real momentum balance + convection). §11.

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

const CACHE_DIR4 = joinpath(@__DIR__, "output_stage4", "cache")
const CACHE_DIR3b = joinpath(@__DIR__, "output_stage3", "cache")

@printf("=== Stage 4 ===\n")
@printf("N_CL=%d, N_GDL=%d\n", MESH_N_CL, MESH_N_GDL)
@printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV, HER_pf = %.3e, alpha_c = %.4f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
@printf("kappa_GDL=%.3e m^2, kappa_CL=%.3e m^2, kappa_interface=%.3e m^2\n", kappa_GDL, kappa_CL, kappa_interface)
@printf("Cache dir: %s\n", CACHE_DIR4)

cl_mesh = make_cl_mesh(MESH_N_CL)
gdl_mesh = make_gdl_mesh(MESH_N_GDL)
ly4 = Layout14(MESH_N_CL, MESH_N_GDL, true)
c_eq_vec = equilibrium_c_vec()

u0 = make_u0_stage34(ly4, c_eq_vec)
let F0 = zeros(ly4.N_TOT)
    residual_stage4!(F0, u0, V_RHE_grid[1], cl_mesh, gdl_mesh, c_eq_vec, ly4)
end

res_at_V! = (F, u, V) -> residual_stage4!(F, u, V, cl_mesh, gdl_mesh, c_eq_vec, ly4)

t0 = time()
Vs, sols = run_continuation!(res_at_V!, u0, V_RHE_grid[1], collect(V_RHE_grid), ly4, 5*MESH_N_CL,
                              CACHE_DIR4, "s4")
elapsed = time() - t0

@printf("\n=== Stage 4 summary ===\nConverged: %d/%d   wall=%.2fs\n", length(Vs), length(V_RHE_grid), elapsed)

println("\n--- Physicality checks ---")
i_CO_arr = Float64[]; i_H2_arr = Float64[]; i_CO_arr3 = Float64[]; i_H2_arr3 = Float64[]
dp_arr = Float64[]
for (k, V) in enumerate(Vs)
    u = sols[k]
    V_SHE = V - V_RHE_to_SHE_offset
    phi_s_of, phi_L_of = const_phi_closures(V_SHE)
    push!(i_CO_arr, compute_i_CO(u, cl_mesh, phi_s_of, phi_L_of)/10.0)
    push!(i_H2_arr, compute_i_H2(u, cl_mesh, phi_s_of, phi_L_of)/10.0)
    p_gdl1 = gdl_p(u, ly4, 1); p_clN = cl_p(u, ly4, MESH_N_CL)
    push!(dp_arr, p_gdl1 - p_clN)

    u3 = load_solution(CACHE_DIR3b, "s3", V)
    if u3 !== nothing
        push!(i_CO_arr3, compute_i_CO(u3, cl_mesh, phi_s_of, phi_L_of)/10.0)
        push!(i_H2_arr3, compute_i_H2(u3, cl_mesh, phi_s_of, phi_L_of)/10.0)
    end
end
@printf("Pressure drop (GDL inlet - CL exit) range: [%.2f, %.2f] Pa  (guide: few hundred Pa to ~1 kPa, rising with current)\n",
        minimum(dp_arr), maximum(dp_arr))
@printf("dp at V_RHE=-1.200,-1.650,-2.075,-2.500 (nearest grid): ")
for Vtest in [-1.200, -1.650, -2.075, -2.500]
    idx = argmin(abs.(Vs .- Vtest))
    @printf("%.1f Pa @ %.3f  ", dp_arr[idx], Vs[idx])
end
println()

if !isempty(i_CO_arr3)
    rel_diff_CO = maximum(abs.(i_CO_arr .- i_CO_arr3) ./ max.(i_CO_arr3, 1e-6))
    rel_diff_H2 = maximum(abs.(i_H2_arr .- i_H2_arr3) ./ max.(i_H2_arr3, 1e-6))
    @printf("Max relative diff Stage4 vs Stage3 currents: iCO=%.4f  iH2=%.4f  (guide: 'almost identical')\n",
            rel_diff_CO, rel_diff_H2)
end

@printf("Peak FE_CO = %.1f%% at V_RHE=%.3f\n", maximum(i_CO_arr./(i_CO_arr.+i_H2_arr)*100),
        Vs[argmax(i_CO_arr./(i_CO_arr.+i_H2_arr))])

write_polarization_csv(joinpath(@__DIR__, "output_stage4", "polarization.csv"), Vs, sols, cl_mesh, stage14_phi_of)
write_liquid_profiles_csv(joinpath(@__DIR__, "output_stage4", "liquid_profiles.csv"), Vs, sols, cl_mesh, stage14_phi_of)
write_gas_profiles_csv(joinpath(@__DIR__, "output_stage4", "gas_profiles.csv"), Vs, sols, gdl_mesh, cl_mesh, ly4)
println("\nStage 4 driver done.")
