# run_stage3.jl — Stage 3: Stefan-Maxwell gas transport (+ volumetric CO2 phase transfer), u_G=0.
# §11.

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

const CACHE_DIR3 = joinpath(@__DIR__, "output_stage3", "cache")

@printf("=== Stage 3 ===\n")
@printf("N_CL=%d, N_GDL=%d\n", MESH_N_CL, MESH_N_GDL)
@printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV, HER_pf = %.3e, alpha_c = %.4f, k_GL=%.2e\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER(), k_GL())
@printf("Cache dir: %s\n", CACHE_DIR3)

cl_mesh = make_cl_mesh(MESH_N_CL)
gdl_mesh = make_gdl_mesh(MESH_N_GDL)
ly3 = Layout14(MESH_N_CL, MESH_N_GDL, true)
c_eq_vec = equilibrium_c_vec()
@printf("Layout: N_TOT = %d (expect %d)\n", ly3.N_TOT, 5*MESH_N_CL + 4*(MESH_N_GDL+MESH_N_CL))

u0 = make_u0_stage34(ly3, c_eq_vec)

# Gas-only JIT warm-up (§11 Stage 3 bootstrap): call the residual once before the real solve to
# trigger compilation on a throwaway call (see BUILD_DIALOGUE.md — a literal "zero liquid source"
# code path was judged unnecessary since compilation cost is independent of N; a single throwaway
# residual call captures the same benefit).
let F0 = zeros(ly3.N_TOT)
    residual_stage3!(F0, u0, V_RHE_grid[1], cl_mesh, gdl_mesh, c_eq_vec, ly3)
end
println("JIT warm-up call complete.")

res_at_V! = (F, u, V) -> residual_stage3!(F, u, V, cl_mesh, gdl_mesh, c_eq_vec, ly3)

t0 = time()
Vs, sols = run_continuation!(res_at_V!, u0, V_RHE_grid[1], collect(V_RHE_grid), ly3, 5*MESH_N_CL,
                              CACHE_DIR3, "s3")
elapsed = time() - t0

@printf("\n=== Stage 3 summary ===\nConverged: %d/%d   wall=%.2fs\n", length(Vs), length(V_RHE_grid), elapsed)

println("\n--- Physicality checks ---")
u_hi = sols[end]
omega_gdl1 = gdl_omega(u_hi, ly3, 1)
omega_cl1  = cl_omega(u_hi, ly3, 1)
omega_clN  = cl_omega(u_hi, ly3, MESH_N_CL)
@printf("At V_RHE=%.3f: omega_CO2 at GDL inlet=%.4f (expect ~0.947)\n", Vs[end], omega_gdl1[1])
@printf("                omega_CO2 at CL cell1=%.4f, CL cell N=%.4f (expect decreasing)\n", omega_cl1[1], omega_clN[1])
@printf("                omega_CO at CL cell1=%.4f, CL cell N=%.4f (expect increasing through CL)\n", omega_cl1[2], omega_clN[2])
@printf("                omega_H2 at CL cell1=%.4f, CL cell N=%.4f (expect increasing through CL)\n", omega_cl1[3], omega_clN[3])
p_cl1 = cl_p(u_hi, ly3, 1); p_clN = cl_p(u_hi, ly3, MESH_N_CL)
@printf("                p(CL cell1)=%.2f Pa, p(CL cellN)=%.2f Pa (Stage3: expect BOTH == p_total=%.0f exactly)\n",
        p_cl1, p_clN, p_total)
@assert abs(p_cl1 - p_total) < 1e-6 && abs(p_clN - p_total) < 1e-6

i_CO_arr = Float64[]; i_H2_arr = Float64[]
for (k, V) in enumerate(Vs)
    V_SHE = V - V_RHE_to_SHE_offset
    phi_s_of, phi_L_of = const_phi_closures(V_SHE)
    push!(i_CO_arr, compute_i_CO(sols[k], cl_mesh, phi_s_of, phi_L_of)/10.0)
    push!(i_H2_arr, compute_i_H2(sols[k], cl_mesh, phi_s_of, phi_L_of)/10.0)
end
@printf("Peak FE_CO = %.1f%% at V_RHE=%.3f (expect HIGHER than Stage2 — gas transport relieves the CO2 limitation)\n",
        maximum(i_CO_arr./(i_CO_arr.+i_H2_arr)*100), Vs[argmax(i_CO_arr./(i_CO_arr.+i_H2_arr))])
@printf("i_total at most cathodic V = %.2f mA/cm^2\n", i_CO_arr[end]+i_H2_arr[end])

write_polarization_csv(joinpath(@__DIR__, "output_stage3", "polarization.csv"), Vs, sols, cl_mesh, stage14_phi_of)
write_liquid_profiles_csv(joinpath(@__DIR__, "output_stage3", "liquid_profiles.csv"), Vs, sols, cl_mesh, stage14_phi_of)
write_gas_profiles_csv(joinpath(@__DIR__, "output_stage3", "gas_profiles.csv"), Vs, sols, gdl_mesh, cl_mesh, ly3)
println("\nStage 3 driver done.")
