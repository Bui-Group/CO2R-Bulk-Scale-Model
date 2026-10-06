# run_stage5.jl — Stage 5: full electrochemistry (phi_L NP migration + phi_s solid Ohm's law),
# using FITTED kinetics from the Fit step. §11 Stage 5.

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
include(joinpath(@__DIR__, "stage5_diagnostics.jl"))

using Printf

# ── Load fitted kinetics (gotcha #14: Stage 5 MUST use fitted, not default, kinetics) ──
const FITTED_PARAMS_FILE = joinpath(@__DIR__, "output_fit", "fitted_params.txt")
isfile(FITTED_PARAMS_FILE) || error("No fitted kinetics found at $FITTED_PARAMS_FILE — run run_fit.jl first!")
for line in eachline(FITTED_PARAMS_FILE)
    isempty(strip(line)) && continue
    k, v = split(line, "="; limit=2)
    val = parse(Float64, v)
    k == "i0_CO2R"       && set_i0_CO2R!(val)
    k == "lambda_MHC"    && set_lambda_MHC!(val)
    k == "HER_prefactor" && set_HER_prefactor!(val)
    k == "alpha_c_HER"   && set_alpha_c_HER!(val)
end

const CACHE_DIR5 = joinpath(@__DIR__, "output_stage5", "cache")
const CACHE_DIR4s = joinpath(@__DIR__, "output_stage4", "cache")
const CACHE_DIRFITs = joinpath(@__DIR__, "output_fit", "cache")

@printf("=== Stage 5 ===\n")
@printf("N_CL=%d, N_GDL=%d\n", MESH_N_CL, MESH_N_GDL)
@printf("Loaded FITTED kinetics: i0_CO2R=%.3f A/m^2, lambda_MHC=%.4f eV, HER_pf=%.4e, alpha_c=%.5f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
@printf("Cache dir: %s\n", CACHE_DIR5)

cl_mesh = make_cl_mesh(MESH_N_CL)
gdl_mesh = make_gdl_mesh(MESH_N_GDL)
ly5 = Stage5Layout(MESH_N_CL, MESH_N_GDL)
c_eq_vec = equilibrium_c_vec()
@printf("Layout: N_TOT = %d (expect %d)\n", ly5.N_TOT,
        5*MESH_N_CL + 4*(MESH_N_GDL+MESH_N_CL) + MESH_N_GDL + MESH_N_CL + MESH_N_CL)

# V_c=0 anchor, built from physics only -- NEVER from a Stage-4/Fit cache (gotcha #0/#9).
u0 = make_u0_stage5_anchor(ly5, c_eq_vec)
let F0 = zeros(ly5.N_TOT)
    residual_stage5!(F0, u0, 0.0, cl_mesh, gdl_mesh, c_eq_vec, ly5)
end
println("JIT warm-up call complete.")

res_at_V! = (F, u, V) -> residual_stage5!(F, u, V, cl_mesh, gdl_mesh, c_eq_vec, ly5)

t0 = time()
Vs, sols = run_stage5_sweep!(res_at_V!, u0, ly5, CACHE_DIR5, "s5")
elapsed = time() - t0

@printf("\n=== Stage 5 summary ===\nConverged (canonical 89-grid): %d/%d   wall=%.2fs\n",
        length(Vs), length(V_RHE_grid), elapsed)

# ── Physicality checks (§9.5.8, §11 Stage 5 checks 1-7) ──
println("\n--- Stage 5 physicality checks ---")
i_CO_arr = Float64[]; i_H2_arr = Float64[]
dphiL_arr = Float64[]; totaldrop_arr = Float64[]; gdlshare_arr = Float64[]
relerr_current_arr = Float64[]
for (k, V) in enumerate(Vs)
    u = sols[k]
    phi_of = make_stage5_phi_of(ly5)
    V_SHE = V - V_RHE_to_SHE_offset
    phi_s_of, phi_L_of = phi_of(u, V_SHE)
    iCO = compute_i_CO(u, cl_mesh, phi_s_of, phi_L_of); iH2 = compute_i_H2(u, cl_mesh, phi_s_of, phi_L_of)
    push!(i_CO_arr, iCO/10.0); push!(i_H2_arr, iH2/10.0)

    diag = stage5_diagnostics(u, ly5, cl_mesh, gdl_mesh, c_eq_vec, V)
    push!(dphiL_arr, diag.dphi_L * 1000)   # mV
    push!(totaldrop_arr, diag.total_solid_drop * 1000)
    push!(gdlshare_arr, diag.gdl_drop / diag.total_solid_drop * 100)

    rel_i = abs(abs(diag.i_far_total) - abs(diag.i_s_face1)) / abs(diag.i_far_total)
    rel_ionic = abs(abs(diag.i_far_total) - abs(diag.i_ionic_R)) / abs(diag.i_far_total)
    push!(relerr_current_arr, rel_i)
    if k == 1 || k == length(Vs) || V ≈ -1.200 || V ≈ -2.425
        @printf("  V_RHE=%.3f: i_far=%.4f i_s[1]=%.4f (rel|Δi|=%.2e)  i_ionic_R=%.4f (rel=%.2e)  ΔφL=%.2fmV  drop=%.2fmV (GDL %.0f%%)\n",
                V, diag.i_far_total, diag.i_s_face1, rel_i, diag.i_ionic_R, rel_ionic,
                diag.dphi_L*1000, diag.total_solid_drop*1000, diag.gdl_drop/diag.total_solid_drop*100)
    end
end
@printf("\nMax relative |Δi| (collector current vs faradaic) over all 89 voltages: %.3e (target <= 1e-6)\n", maximum(relerr_current_arr))

i_tot_arr = i_CO_arr .+ i_H2_arr
hi_idx = findall(i_tot_arr .> 50.0)   # i_total > 500 mA/cm^2 -> here in mA/cm^2 so > 500... wait unit check below
hi_idx = findall(i_tot_arr .> 500.0)
if !isempty(hi_idx)
    @printf("Delta phi_L range where i_total > 500 mA/cm^2: [%.2f, %.2f] mV (target 5-25 mV)\n",
            minimum(abs.(dphiL_arr[hi_idx])), maximum(abs.(dphiL_arr[hi_idx])))
end
@printf("Total solid ohmic drop range: [%.2f, %.2f] mV; GDL share: [%.1f, %.1f]%% (target ~80-90%%)\n",
        minimum(totaldrop_arr), maximum(totaldrop_arr), minimum(gdlshare_arr), maximum(gdlshare_arr))

# Smoothness / monotonicity
di_CO = diff(i_CO_arr)
sign_ch_CO = count(k -> sign(di_CO[k]) != sign(di_CO[k+1]) && di_CO[k]!=0 && di_CO[k+1]!=0, 1:length(di_CO)-1)
di_H2 = diff(i_H2_arr)
sign_ch_H2 = count(k -> sign(di_H2[k]) != sign(di_H2[k+1]) && di_H2[k]!=0 && di_H2[k+1]!=0, 1:length(di_H2)-1)
@printf("Sign changes in di_CO/dV: %d (target: exactly 1)   di_H2/dV: %d (target: 0)\n", sign_ch_CO, sign_ch_H2)

# Compare against the Fit-step's fitted-kinetics Stage-4 sweep (same kinetics, no potentials)
i_CO_fit = Float64[]; i_H2_fit = Float64[]
for V in Vs
    u4 = load_solution(CACHE_DIRFITs, "fit", V)
    if u4 !== nothing
        V_SHE = V - V_RHE_to_SHE_offset
        phi_s_of, phi_L_of = const_phi_closures(V_SHE)
        push!(i_CO_fit, compute_i_CO(u4, cl_mesh, phi_s_of, phi_L_of)/10.0)
        push!(i_H2_fit, compute_i_H2(u4, cl_mesh, phi_s_of, phi_L_of)/10.0)
    end
end
if length(i_CO_fit) == length(Vs)
    rel_H2 = (i_H2_fit .- i_H2_arr) ./ i_H2_fit .* 100
    rel_CO = (i_CO_arr .- i_CO_fit) ./ i_CO_fit .* 100
    idx_hi = argmax(i_tot_arr)
    @printf("At the highest-current voltage (V_RHE=%.3f): i_H2 change vs fitted-Stage4 = %.1f%% (target: falls 3-10%%)\n",
            Vs[idx_hi], -rel_H2[idx_hi])
    @printf("                                              i_CO change vs fitted-Stage4 = %.1f%% (may fall OR rise)\n",
            rel_CO[idx_hi])
end

peakFE, peakidx = findmax(i_CO_arr ./ i_tot_arr .* 100)
@printf("Peak FE_CO = %.1f%% at V_RHE=%.3f\n", peakFE, Vs[peakidx])

mkpath(joinpath(@__DIR__, "output_stage5"))
write_polarization_csv(joinpath(@__DIR__, "output_stage5", "polarization.csv"), Vs, sols, cl_mesh, make_stage5_phi_of(ly5))
write_liquid_profiles_csv(joinpath(@__DIR__, "output_stage5", "liquid_profiles.csv"), Vs, sols, cl_mesh, make_stage5_phi_of(ly5))
write_gas_profiles_csv(joinpath(@__DIR__, "output_stage5", "gas_profiles.csv"), Vs, sols, gdl_mesh, cl_mesh, ly5)
write_potential_profiles_csv(joinpath(@__DIR__, "output_stage5", "phi_s_profile.csv"),
                              joinpath(@__DIR__, "output_stage5", "phi_L_profile.csv"),
                              Vs, sols, gdl_mesh, cl_mesh, ly5)
println("\nStage 5 driver done.")
