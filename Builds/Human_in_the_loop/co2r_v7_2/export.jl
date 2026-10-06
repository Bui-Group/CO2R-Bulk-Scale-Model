# =============================================================================
# export.jl — dump sweep results for the Python plotting layer (§12.7, §18).
# Plain CSV for the polarization data; raw Float64 binaries for the profile
# figures (same philosophy as the §10.3 solution cache: no serialization lib).
# =============================================================================
using Printf

outdir_for(stage_tag::String) = joinpath(@__DIR__, "output_$(stage_tag)")

"Polarization CSV: one row per swept voltage."
function export_polarization(stage_tag::String, Vs::Vector{Float64},
                             sols::Vector{Vector{Float64}}, ctx::ModelCtx)
    od = outdir_for(stage_tag); mkpath(od)
    fn = joinpath(od, "polarization.csv")
    open(fn, "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO_pct,FE_H2_pct")
        for (k, V) in enumerate(Vs)
            iCO, iH2, itot = compute_currents(sols[k], V, ctx)
            iCO /= 10.0; iH2 /= 10.0; itot /= 10.0          # A/m^2 -> mA/cm^2
            feCO = itot > 0 ? 100 * iCO / itot : 0.0
            @printf(f, "%.6f,%.6f,%.8e,%.8e,%.8e,%.6f,%.6f\n",
                    V, V - V_RHE_to_SHE_offset, iCO, iH2, itot, feCO, 100 - feCO)
        end
    end
    return fn
end

"3x3 liquid-phase profile figure data (§12.7 A2): 9 fields x N_CL x n_V."
function export_liquid_profiles(stage_tag::String, Vs::Vector{Float64},
                                sols::Vector{Vector{Float64}}, ctx::ModelCtx)
    od = outdir_for(stage_tag); mkpath(od)
    fn = joinpath(od, "profiles_liquid.bin")
    ly = ctx.ly; Nc = ly.N_CL
    open(fn, "w") do f
        write(f, Int64(length(Vs))); write(f, Int64(Nc))
        write(f, collect(ctx.cl.x_cent))
        write(f, collect(Vs))
        for (k, V) in enumerate(Vs)
            u = sols[k]
            c = unpack_log_conc(u, ly)
            rc, rh = compute_rates(u, V, ctx)
            cCs = [cCs_from_electroneutrality(view(c, :, ix)) for ix in 1:Nc]
            write(f, [-log10(c[1, ix] / 1000.0) for ix in 1:Nc])   # (a) pH
            write(f, collect(c[5, :]))                             # (b) CO2(aq)
            write(f, collect(c[2, :]))                             # (c) OH-
            write(f, collect(c[3, :]))                             # (d) HCO3-
            write(f, collect(c[4, :]))                             # (e) CO3^2-
            write(f, cCs)                                          # (f) Cs+
            write(f, collect(c[1, :]))                             # (g) H+
            write(f, rc)                                           # (h) R_CO2R
            write(f, rh)                                           # (i) R_HER
        end
    end
    return fn
end

"2x2 gas-phase profile figure data (§12.7 B1): 4 fields x (N_GDL+N_CL) x n_V."
function export_gas_profiles(stage_tag::String, Vs::Vector{Float64},
                             sols::Vector{Vector{Float64}}, ctx::ModelCtx)
    ctx.ly.has_gas || return ""
    od = outdir_for(stage_tag); mkpath(od)
    fn = joinpath(od, "profiles_gas.bin")
    ly = ctx.ly; Ng = ly.N_GDL; Nc = ly.N_CL; Ntot = Ng + Nc
    open(fn, "w") do f
        write(f, Int64(length(Vs))); write(f, Int64(Ntot)); write(f, Int64(Ng))
        write(f, collect(ctx.gg.x_cent))
        write(f, collect(Vs))
        for (k, _) in enumerate(Vs)
            u = sols[k]
            pC = zeros(Ntot); pO = zeros(Ntot); pH = zeros(Ntot); pT = zeros(Ntot)
            for m in 1:Ntot
                w, p = gas_cell_state(u, ly, m)
                x = omega_to_x(w)
                pC[m] = x[1] * p / 101325.0
                pO[m] = x[2] * p / 101325.0
                pH[m] = x[3] * p / 101325.0
                pT[m] = p
            end
            write(f, pC); write(f, pO); write(f, pH); write(f, pT)
        end
    end
    return fn
end

"phi_s (GDL+CL) and phi_L (CL) profile data (§12.7 D1/D2)."
function export_potential_profiles(stage_tag::String, Vs::Vector{Float64},
                                   sols::Vector{Vector{Float64}}, ctx::ModelCtx)
    ctx.ly.has_pot || return ""
    od = outdir_for(stage_tag); mkpath(od)
    fn = joinpath(od, "profiles_potential.bin")
    ly = ctx.ly; Ng = ly.N_GDL; Nc = ly.N_CL
    open(fn, "w") do f
        write(f, Int64(length(Vs))); write(f, Int64(Ng)); write(f, Int64(Nc))
        write(f, vcat(collect(ctx.gdl.x_cent), collect(ctx.cl.x_cent)))
        write(f, collect(ctx.cl.x_cent))
        write(f, collect(Vs))
        for (k, _) in enumerate(Vs)
            u = sols[k]
            write(f, vcat(unpack_phi_s_gdl(u, ly), unpack_phi_s_cl(u, ly)))
            write(f, unpack_phi_L_cl(u, ly))
        end
    end
    return fn
end

"Everything a stage needs for its figures."
function export_stage(stage_tag::String, Vs::Vector{Float64},
                      sols::Vector{Vector{Float64}}, ctx::ModelCtx)
    fs = String[]
    push!(fs, export_polarization(stage_tag, Vs, sols, ctx))
    push!(fs, export_liquid_profiles(stage_tag, Vs, sols, ctx))
    g = export_gas_profiles(stage_tag, Vs, sols, ctx);       isempty(g) || push!(fs, g)
    p = export_potential_profiles(stage_tag, Vs, sols, ctx); isempty(p) || push!(fs, p)
    return fs
end
