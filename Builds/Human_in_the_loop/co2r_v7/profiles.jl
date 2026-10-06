# =====================================================================
#  profiles.jl — export spatial profiles for the §12.7 debug plots
# =====================================================================
using Printf

"A2 (3×3 liquid) data: one row per (voltage, CL cell)."
function write_liquid_profiles(dir, records::Vector{StepRecord}, cache_dir, prefix,
                               cl::Mesh1D, ly::ModelLayout)
    mkpath(dir)
    open(joinpath(dir, "profiles_liquid.csv"), "w") do io
        println(io, "V_rhe,V_she,x_um,pH,c_CO2,c_OH,c_HCO3,c_CO3,c_Cs,c_H,R_CO2R,R_HER")
        for r in records
            u = load_solution(cache_dir, prefix, r.V_rhe); u === nothing && continue
            c = unpack_log_conc(u, cl.N)
            V_applied_SHE = r.V_rhe - V_RHE_to_SHE_offset
            for ix in 1:cl.N
                ps = ly.stage == 5 ? u[phi_s_cl_idx5(ly, ix)] : V_applied_SHE
                pl = ly.stage == 5 ? u[phi_L_idx5(ly, ix)]    : 0.0
                r1 = R_CO2R(c[5,ix], ps, pl, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
                r2 = R_HER(ps, pl, c[1,ix], a_v_CL, n_e)
                cCs = cCs_from_electroneutrality(view(c, :, ix))
                @printf(io, "%.6f,%.6f,%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        r.V_rhe, r.V_she, cl.x_centers[ix]*1e6,
                        -log10(max(c[1,ix]/1000, 1e-30)),
                        c[5,ix], c[2,ix], c[3,ix], c[4,ix], cCs, c[1,ix], r1, r2)
            end
        end
    end
end

"B1 (2×2 gas) data: one row per (voltage, gas cell) across GDL+CL."
function write_gas_profiles(dir, records::Vector{StepRecord}, cache_dir, prefix,
                            cl::Mesh1D, gdl::Mesh1D, ly::ModelLayout)
    ly.stage >= 3 || return
    mkpath(dir)
    xg = vcat(gdl.x_centers, cl.x_centers)
    open(joinpath(dir, "profiles_gas.csv"), "w") do io
        println(io, "V_rhe,V_she,x_um,region,p_CO2_atm,p_CO_atm,p_H2_atm,p_H2O_atm,p_total_Pa,omega_CO2")
        for r in records
            u = load_solution(cache_dir, prefix, r.V_rhe); u === nothing && continue
            for g in 1:(ly.N_GDL + ly.N_CL)
                w = gas_omega(u, ly, g); p = gas_p(u, ly, g)
                Mn = Mn_from_omega(w); x = omega_to_x(w, Mn)
                @printf(io, "%.6f,%.6f,%.6f,%s,%.8e,%.8e,%.8e,%.8e,%.6f,%.8e\n",
                        r.V_rhe, r.V_she, xg[g]*1e6, g <= ly.N_GDL ? "GDL" : "CL",
                        x[1]*p/101325, x[2]*p/101325, x[3]*p/101325, x[4]*p/101325, p, w[1])
            end
        end
    end
end

"D1/D2 (Stage 5 potentials): φ_s across GDL+CL, φ_L across CL."
function write_potential_profiles(dir, records::Vector{StepRecord}, cache_dir, prefix,
                                  cl::Mesh1D, gdl::Mesh1D, ly::ModelLayout)
    ly.stage == 5 || return
    mkpath(dir)
    open(joinpath(dir, "profiles_phi.csv"), "w") do io
        println(io, "V_rhe,V_she,x_um,region,phi_s,phi_L")
        for r in records
            u = load_solution(cache_dir, prefix, r.V_rhe); u === nothing && continue
            for ig in 1:ly.N_GDL
                @printf(io, "%.6f,%.6f,%.6f,GDL,%.8e,\n", r.V_rhe, r.V_she,
                        gdl.x_centers[ig]*1e6, u[phi_s_gdl_idx5(ly, ig)])
            end
            for ix in 1:ly.N_CL
                @printf(io, "%.6f,%.6f,%.6f,CL,%.8e,%.8e\n", r.V_rhe, r.V_she,
                        cl.x_centers[ix]*1e6, u[phi_s_cl_idx5(ly, ix)], u[phi_L_idx5(ly, ix)])
            end
        end
    end
end

function export_all_profiles(dir, records, cache_dir, prefix, cl, gdl, ly)
    write_liquid_profiles(dir, records, cache_dir, prefix, cl, ly)
    write_gas_profiles(dir, records, cache_dir, prefix, cl, gdl, ly)
    write_potential_profiles(dir, records, cache_dir, prefix, cl, gdl, ly)
end
