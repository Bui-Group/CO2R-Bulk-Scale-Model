##############################################################################
# postprocess.jl — diagnostics + CSV export shared by all run_stageN.jl (§12)
##############################################################################
using Printf

function liquid_diagnostics_simple(u::Vector{Float64}, cl_mesh::Mesh1D, V_applied_SHE::Float64)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    pH = [-log10(max(c[1,ix]/1000, 1e-20)) for ix in 1:N]
    cCs = [c_Cs_of(@view c[:,ix]) for ix in 1:N]
    rco2r = [R_CO2R(c[5,ix], V_applied_SHE, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N]
    rher  = [R_HER(V_applied_SHE, 0.0, c[1,ix], a_v_CL, n_e) for ix in 1:N]
    return (pH=pH, cH=c[1,:], cOH=c[2,:], cHCO3=c[3,:], cCO3=c[4,:], cCO2=c[5,:], cCs=cCs,
            R_CO2R=rco2r, R_HER=rher)
end

function liquid_diagnostics_stage5(u::Vector{Float64}, cl_mesh::Mesh1D, ly::Stage5Layout)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    phi_s_cl = unpack_phi_s_cl(u, ly); phi_L = unpack_phi_L(u, ly)
    pH = [-log10(max(c[1,ix]/1000, 1e-20)) for ix in 1:N]
    cCs = [c_Cs_of(@view c[:,ix]) for ix in 1:N]
    rco2r = [R_CO2R(c[5,ix], phi_s_cl[ix], phi_L[ix], c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N]
    rher  = [R_HER(phi_s_cl[ix], phi_L[ix], c[1,ix], a_v_CL, n_e) for ix in 1:N]
    return (pH=pH, cH=c[1,:], cOH=c[2,:], cHCO3=c[3,:], cCO3=c[4,:], cCO2=c[5,:], cCs=cCs,
            R_CO2R=rco2r, R_HER=rher)
end

function gas_diagnostics(u::Vector{Float64}, ly::GasCapableLayout, gdl_mesh::Mesh1D, cl_mesh::Mesh1D)
    N_GDL = gdl_mesh.N; N_CL = cl_mesh.N
    xs = vcat(gdl_x_global(gdl_mesh), cl_x_global(cl_mesh))
    pCO2 = zeros(N_GDL+N_CL); pCO = zeros(N_GDL+N_CL); pH2 = zeros(N_GDL+N_CL); ptot = zeros(N_GDL+N_CL)
    for ig in 1:N_GDL
        w1,w2,w3,p = gdl_gas(u, ly, ig)
        Mn = Mn_from_omega(w1,w2,w3)
        x1,x2,x3,x4 = omega_to_x(w1,w2,w3,Mn)
        pCO2[ig]=x1*p/101325; pCO[ig]=x2*p/101325; pH2[ig]=x3*p/101325; ptot[ig]=p
    end
    for ix in 1:N_CL
        w1,w2,w3,p = cl_gas(u, ly, ix)
        Mn = Mn_from_omega(w1,w2,w3)
        x1,x2,x3,x4 = omega_to_x(w1,w2,w3,Mn)
        idx = N_GDL+ix
        pCO2[idx]=x1*p/101325; pCO[idx]=x2*p/101325; pH2[idx]=x3*p/101325; ptot[idx]=p
    end
    return (x_um=xs.*1e6, p_CO2=pCO2, p_CO=pCO, p_H2=pH2, p_total=ptot)
end

function export_polarization_csv(path::String, Vs_RHE::Vector{Float64}, iCO_mA::Vector{Float64}, iH2_mA::Vector{Float64})
    open(path, "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mA_cm2,i_H2_mA_cm2,i_total_mA_cm2,FE_CO_pct,FE_H2_pct")
        for k in eachindex(Vs_RHE)
            V_rhe = Vs_RHE[k]; V_she = V_rhe - V_RHE_to_SHE_offset
            ico = iCO_mA[k]; ih2 = iH2_mA[k]; itot = ico+ih2
            feco = itot > 0 ? 100*ico/itot : 0.0
            feh2 = itot > 0 ? 100*ih2/itot : 0.0
            @printf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f\n", V_rhe, V_she, ico, ih2, itot, feco, feh2)
        end
    end
end

function export_liquid_profiles_csv(path::String, cl_mesh::Mesh1D, Vs_RHE::Vector{Float64}, rows_fn::Function)
    open(path, "w") do f
        println(f, "V_RHE,x_um,pH,c_CO2,c_OH,c_HCO3,c_CO3,c_Cs,c_H,R_CO2R,R_HER")
        for V in Vs_RHE
            d = rows_fn(V)
            for ix in 1:cl_mesh.N
                @printf(f, "%.6f,%.6f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                        V, cl_mesh.x_center[ix]*1e6, d.pH[ix], d.cCO2[ix], d.cOH[ix], d.cHCO3[ix],
                        d.cCO3[ix], d.cCs[ix], d.cH[ix], d.R_CO2R[ix], d.R_HER[ix])
            end
        end
    end
end

function export_gas_profiles_csv(path::String, Vs_RHE::Vector{Float64}, rows_fn::Function, N_tot::Int)
    open(path, "w") do f
        println(f, "V_RHE,x_um,p_CO2_atm,p_CO_atm,p_H2_atm,p_total_Pa")
        for V in Vs_RHE
            d = rows_fn(V)
            for k in 1:N_tot
                @printf(f, "%.6f,%.6f,%.6e,%.6e,%.6e,%.6e\n",
                        V, d.x_um[k], d.p_CO2[k], d.p_CO[k], d.p_H2[k], d.p_total[k])
            end
        end
    end
end

"""
    stage5_diagnostics(u, ly, cl_mesh, gdl_mesh, V_cathode, c_eq_vec)

Post-solve physicality diagnostics for Stage 5 (§11 Stage 5 checks 1,2,4,6).
"""
function stage5_diagnostics(u::Vector{Float64}, ly::Stage5Layout, cl_mesh::Mesh1D,
                             gdl_mesh::Mesh1D, V_cathode::Float64, c_eq_vec::Vector{Float64})
    N_CL = cl_mesh.N; N_GDL = gdl_mesh.N
    c = unpack_log_conc(u, N_CL)
    phi_s_gdl = unpack_phi_s_gdl(u, ly); phi_s_cl = unpack_phi_s_cl(u, ly); phi_L = unpack_phi_L(u, ly)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    i_s = solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, sigma_s_eff_GDL, sigma_s_eff_CL, gdl_mesh, cl_mesh)
    i_s_face1 = i_s[1]

    i_far_total = 0.0
    for ix in 1:N_CL
        c_ix = @view c[:,ix]
        r_co2r = R_CO2R(c_ix[5], phi_s_cl[ix], phi_L[ix], c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(phi_s_cl[ix], phi_L[ix], c_ix[1], a_v_CL, n_e)
        i_far_total += n_e*F_const*(r_co2r+r_her)*cl_mesh.dx[ix]
    end

    # ionic current at the right (electrolyte) boundary, WITH migration; Cs+ flux is 0 there by construction (§3.1)
    i_ionic_R = 0.0
    for i in 1:5
        L_eff = D_eff_vec[i]/k_MT_vec[i]
        Jf = np_flux(c[i,N_CL], c_eq_vec[i], phi_L[N_CL], 0.0, D_eff_vec[i], z_vec[i], L_eff)
        i_ionic_R += z_vec[i]*Jf
    end
    i_ionic_R *= F_const

    total_solid_drop = phi_s_cl[N_CL] - V_applied_SHE
    gdl_drop = phi_s_gdl[N_GDL] - V_applied_SHE
    cl_drop = phi_s_cl[N_CL] - phi_s_gdl[N_GDL]

    return (i_s_face1=i_s_face1, i_far_total=i_far_total, i_ionic_R=i_ionic_R,
            dphiL_mV=(phi_L[1]-phi_L[N_CL])*1000,
            total_solid_drop_mV=total_solid_drop*1000, gdl_drop_mV=gdl_drop*1000, cl_drop_mV=cl_drop*1000)
end

function export_phi_profiles_csv(path_s::String, path_L::String, Vs_RHE::Vector{Float64},
                                  gdl_mesh::Mesh1D, cl_mesh::Mesh1D, ly::Stage5Layout, rows_fn::Function)
    xs_s = vcat(gdl_x_global(gdl_mesh), cl_x_global(cl_mesh))
    open(path_s, "w") do f
        println(f, "V_RHE,x_m,phi_s_V_SHE")
        for V in Vs_RHE
            u = rows_fn(V)
            phis = vcat(unpack_phi_s_gdl(u,ly), unpack_phi_s_cl(u,ly))
            for k in eachindex(xs_s)
                @printf(f, "%.6f,%.8e,%.6e\n", V, xs_s[k], phis[k])
            end
        end
    end
    open(path_L, "w") do f
        println(f, "V_RHE,x_frac,phi_L_V_SHE")
        for V in Vs_RHE
            u = rows_fn(V)
            phiL = unpack_phi_L(u, ly)
            for ix in 1:cl_mesh.N
                @printf(f, "%.6f,%.6f,%.6e\n", V, cl_mesh.x_center[ix]/cl_mesh.L, phiL[ix])
            end
        end
    end
end
