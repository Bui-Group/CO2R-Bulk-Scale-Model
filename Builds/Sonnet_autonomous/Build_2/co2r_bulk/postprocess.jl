# postprocess.jl — writes plot-ready CSVs from cached solutions. Python (plot_stage.py) does no
# physics of its own; it only reads these CSVs and calls matplotlib, so there is no risk of the
# Julia and Python layers silently drifting apart on the physics.

using Printf

function write_polarization_csv(path::String, Vs::Vector{Float64}, sols::Vector{Vector{Float64}},
                                 cl_mesh::FVMesh, phi_of::Function)
    open(path, "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mA_cm2,i_H2_mA_cm2,i_tot_mA_cm2,FE_CO,FE_H2")
        for (k, V) in enumerate(Vs)
            u = sols[k]
            V_SHE = V - V_RHE_to_SHE_offset
            phi_s_of, phi_L_of = phi_of(u, V_SHE)
            iCO = compute_i_CO(u, cl_mesh, phi_s_of, phi_L_of) / 10.0
            iH2 = compute_i_H2(u, cl_mesh, phi_s_of, phi_L_of) / 10.0
            FE_CO = iCO / max(iCO+iH2, 1e-12) * 100
            FE_H2 = 100 - FE_CO
            @printf(f, "%.6f,%.6f,%.8f,%.8f,%.8f,%.4f,%.4f\n", V, V_SHE, iCO, iH2, iCO+iH2, FE_CO, FE_H2)
        end
    end
end

"phi_of for Stages 1-4: constant (V_applied_SHE, 0.0) everywhere."
stage14_phi_of(u, V_SHE) = const_phi_closures(V_SHE)

"phi_of for Stage 5: read the phi_s_cl/phi_L DOFs (u is the Stage-5 vector, ly captured by closure)."
make_stage5_phi_of(ly::Stage5Layout) = (u, V_SHE) -> begin
    phi_s_cl = unpack_phi_s_cl5(u, ly)
    phi_L    = unpack_phi_L_cl5(u, ly)
    ((ix -> phi_s_cl[ix]), (ix -> phi_L[ix]))
end

function write_liquid_profiles_csv(path::String, Vs::Vector{Float64}, sols::Vector{Vector{Float64}},
                                    cl_mesh::FVMesh, phi_of::Function)
    N = cl_mesh.N
    open(path, "w") do f
        println(f, "V_RHE,x_um,pH,cH,cOH,cHCO3,cCO3,cCs,cCO2,R_CO2R,R_HER")
        for (k, V) in enumerate(Vs)
            u = sols[k]
            V_SHE = V - V_RHE_to_SHE_offset
            phi_s_of, phi_L_of = phi_of(u, V_SHE)
            c = unpack_log_conc(u, N)
            for ix in 1:N
                cH, cOH, cHCO3, cCO3, cCO2 = c[1,ix], c[2,ix], c[3,ix], c[4,ix], c[5,ix]
                cCs = c_Cs_from_electroneutrality(cH, cOH, cHCO3, cCO3)
                pH = -log10(cH/1000.0)
                r_co2r = R_CO2R(cCO2, phi_s_of(ix), phi_L_of(ix), cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e_val)
                r_her  = R_HER(phi_s_of(ix), phi_L_of(ix), cH, a_v_CL, n_e_val)
                @printf(f, "%.6f,%.4f,%.4f,%.6e,%.6e,%.4f,%.4f,%.4f,%.4f,%.6e,%.6e\n",
                        V, cl_mesh.x_centers[ix]*1e6, pH, cH, cOH, cHCO3, cCO3, cCs, cCO2, r_co2r, r_her)
            end
        end
    end
end

function write_gas_profiles_csv(path::String, Vs::Vector{Float64}, sols::Vector{Vector{Float64}},
                                 gdl_mesh::FVMesh, cl_mesh::FVMesh, ly)
    N_GDL = gdl_mesh.N; N_CL = cl_mesh.N
    open(path, "w") do f
        println(f, "V_RHE,x_um,domain,p_CO2_atm,p_CO_atm,p_H2_atm,p_total_Pa")
        for (k, V) in enumerate(Vs)
            u = sols[k]
            for ig in 1:N_GDL
                om = gdl_omega(u, ly, ig); p = gdl_p(u, ly, ig)
                x = omega4_to_x4(om)
                @printf(f, "%.6f,%.4f,GDL,%.6f,%.6f,%.6f,%.4f\n",
                        V, gdl_mesh.x_centers[ig]*1e6, x[1]*p/101325.0, x[2]*p/101325.0, x[3]*p/101325.0, p)
            end
            for ix in 1:N_CL
                om = cl_omega(u, ly, ix); p = cl_p(u, ly, ix)
                x = omega4_to_x4(om)
                @printf(f, "%.6f,%.4f,CL,%.6f,%.6f,%.6f,%.4f\n",
                        V, cl_mesh.x_centers[ix]*1e6, x[1]*p/101325.0, x[2]*p/101325.0, x[3]*p/101325.0, p)
            end
        end
    end
end

function write_potential_profiles_csv(path_phis::String, path_phiL::String,
                                       Vs::Vector{Float64}, sols::Vector{Vector{Float64}},
                                       gdl_mesh::FVMesh, cl_mesh::FVMesh, ly::Stage5Layout)
    N_GDL = gdl_mesh.N; N_CL = cl_mesh.N
    open(path_phis, "w") do f
        println(f, "V_RHE,x_m,domain,phi_s")
        for (k, V) in enumerate(Vs)
            u = sols[k]
            phi_s_gdl = unpack_phi_s_gdl5(u, ly); phi_s_cl = unpack_phi_s_cl5(u, ly)
            for ig in 1:N_GDL
                @printf(f, "%.6f,%.8e,GDL,%.6f\n", V, gdl_mesh.x_centers[ig], phi_s_gdl[ig])
            end
            for ix in 1:N_CL
                @printf(f, "%.6f,%.8e,CL,%.6f\n", V, cl_mesh.x_centers[ix], phi_s_cl[ix])
            end
        end
    end
    open(path_phiL, "w") do f
        println(f, "V_RHE,x_over_L,phi_L")
        for (k, V) in enumerate(Vs)
            u = sols[k]
            phi_L = unpack_phi_L_cl5(u, ly)
            for ix in 1:N_CL
                @printf(f, "%.6f,%.6f,%.6f\n", V, cl_mesh.x_centers[ix]/L_CL, phi_L[ix])
            end
        end
    end
end
