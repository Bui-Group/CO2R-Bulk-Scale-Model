# stage5_diag.jl — Stage-5 post-solve diagnostics (§11 Stage 5 checks 1–7, §9.5.8)

"""
    stage5_diagnostics(u, V_cathode, cl, gm, c_eq, ly) -> NamedTuple

Recomputes, from a converged Stage-5 DOF vector, everything the guide asks to
be checked: the φ_L drop, both charge-conservation balances, and the solid
ohmic drop split between GDL and CL.
"""
function stage5_diagnostics(u::Vector{Float64}, V_cathode::Float64,
                            cl::Mesh1D, gm::GasMesh, c_eq::Vector{Float64},
                            ly::Layout)
    N = cl.N
    V_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N)
    phi_L = [u[phi_L_idx(ly, ix)] for ix in 1:N]
    phi_s = [u[phi_s_idx(ly, g)]  for g in 1:gm.N]

    iCO, iH2 = compute_currents5(u, ly, cl)
    i_far = iCO + iH2                                   # A/m², cathodic > 0

    # solid current at the collector face (ghost-cell Dirichlet)
    i_s1 = -sigma_s_eff_GDL * (phi_s[1] - V_SHE) / gm.dxc[1]

    # ionic current at the CL/electrolyte boundary, WITH migration
    i_ionic_R = 0.0
    for i in 1:5
        L_eff = D_eff[i] / k_MT[i]
        J = np_flux(c[i, N], c_eq[i], phi_L[N], 0.0, D_eff[i], z_ion[i], L_eff)
        i_ionic_R += F_const * z_ion[i] * J
    end

    dphi_L = phi_L[1] - phi_L[N]
    phi_s_cl_end = phi_s[gm.N]
    drop_total = phi_s_cl_end - V_SHE                   # rise from collector
    drop_gdl   = phi_s[gm.N_GDL] - V_SHE
    cCs = [c[2,ix] + c[3,ix] + 2*c[4,ix] - c[1,ix] - C_fix*eps_I for ix in 1:N]

    return (V = V_cathode, iCO = iCO, iH2 = iH2, i_far = i_far,
            i_s1 = i_s1, di_rel = abs(abs(i_far) - abs(i_s1)) / abs(i_far),
            i_ionic_R = i_ionic_R,
            di_ionic_rel = abs(abs(i_far) - abs(i_ionic_R)) / abs(i_far),
            phi_L1 = phi_L[1], phi_Lmid = phi_L[max(1, N ÷ 2)], phi_LN = phi_L[N],
            dphi_L = dphi_L,
            drop_total = drop_total, drop_gdl = drop_gdl,
            gdl_share = drop_gdl / drop_total,
            cCs1 = cCs[1], cCs_max = maximum(cCs),
            cCO2_1 = c[5,1], cCO3_1 = c[4,1],
            cOH_avg = sum(c[2,ix]*cl.dx[ix] for ix in 1:N) / L_CL,
            H_eff_1 = sechenov_H_CO2(@view c[:,1]))
end
