# run_stage2.jl — Stage 2: Sherwood-Reynolds Neumann BC at the electrolyte side. §11.

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

const CACHE_DIR2 = joinpath(@__DIR__, "output_stage2", "cache")

@printf("=== Stage 2 ===\n")
@printf("N_CL=%d (production default 80)\n", MESH_N_CL)
@printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV, HER_pf = %.3e, alpha_c = %.4f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
@printf("Cache dir: %s\n", CACHE_DIR2)

cl_mesh = make_cl_mesh(MESH_N_CL)
ly2 = Layout14(MESH_N_CL, 0, false)
c_eq_vec = equilibrium_c_vec()

# Biot number check (§11 Stage 2): Bi_i = k_MT,i * L_CL / D_eff,i, expect O(0.1-10)
println("\n--- Biot numbers (k_MT * L_CL / D_eff) ---")
species_names = ["H+", "OH-", "HCO3-", "CO3^2-", "CO2(aq)"]
for i in 1:5
    Bi = k_MT[i] * L_CL / D_eff[i]
    @printf("  %-7s Bi = %.4f\n", species_names[i], Bi)
end

u0 = make_u0_liquid_eq(MESH_N_CL, c_eq_vec)
res_at_V! = (F, u, V) -> residual_stage2!(F, u, V, cl_mesh, c_eq_vec)

t0 = time()
Vs, sols = run_continuation!(res_at_V!, u0, V_RHE_grid[1], collect(V_RHE_grid), ly2, 5*MESH_N_CL,
                              CACHE_DIR2, "s2")
elapsed = time() - t0

@printf("\n=== Stage 2 summary ===\nConverged: %d/%d   wall=%.2fs\n", length(Vs), length(V_RHE_grid), elapsed)

println("\n--- Physicality checks ---")
c_last_hi = unpack_log_conc(sols[end], MESH_N_CL)   # most cathodic voltage, last cell = electrolyte side
@printf("At V_RHE=%.3f (most cathodic): c_CO2(L_CL)=%.4f mol/m^3 (expect -> depleted, small)\n",
        Vs[end], c_last_hi[5, end])
@printf("                                c_OH(L_CL)=%.4f mol/m^3 (expect >> bulk eq %.4e)\n",
        c_last_hi[2, end], c_eq_vec[2])

flux_OH_right = -k_MT[2] * (c_eq_vec[2] - c_last_hi[2, end])
@printf("Flux check OH- at electrolyte face: %.4e mol/m^2/s (expect positive = leaving CL)\n", flux_OH_right)
flux_CO2_right = -k_MT[5] * (c_eq_vec[5] - c_last_hi[5, end])
@printf("Flux check CO2 at electrolyte face: %.4e mol/m^2/s (expect negative = resupply from reservoir)\n", flux_CO2_right)

i_CO_arr = Float64[]; i_H2_arr = Float64[]
for (k, V) in enumerate(Vs)
    V_SHE = V - V_RHE_to_SHE_offset
    phi_s_of, phi_L_of = const_phi_closures(V_SHE)
    push!(i_CO_arr, compute_i_CO(sols[k], cl_mesh, phi_s_of, phi_L_of)/10.0)
    push!(i_H2_arr, compute_i_H2(sols[k], cl_mesh, phi_s_of, phi_L_of)/10.0)
end
@printf("Peak FE_CO = %.1f%% at V_RHE=%.3f\n", maximum(i_CO_arr./(i_CO_arr.+i_H2_arr)*100),
        Vs[argmax(i_CO_arr./(i_CO_arr.+i_H2_arr))])

write_polarization_csv(joinpath(@__DIR__, "output_stage2", "polarization.csv"), Vs, sols, cl_mesh, stage14_phi_of)
write_liquid_profiles_csv(joinpath(@__DIR__, "output_stage2", "liquid_profiles.csv"), Vs, sols, cl_mesh, stage14_phi_of)
println("\nStage 2 driver done.")
