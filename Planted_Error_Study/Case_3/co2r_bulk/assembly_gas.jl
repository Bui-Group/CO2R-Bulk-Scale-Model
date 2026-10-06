# assembly_gas.jl — Stage 3 & 4 monolithic residual (liquid + Stefan-Maxwell gas + Darcy).
# DOF layout: [ liquid 5×N_CL ; gas GDL 4×N_GDL ; gas CL 4×N_CL ], gas order (ωCO2,ωCO,ωH2,p).

# Gas DOF index in the combined GDL-then-CL ordering.
gas_idx_g(g, j, NCL, NGDL) = g <= NGDL ? (5*NCL + 4*(g-1) + j) :
                                          (5*NCL + 4*NGDL + 4*(g-NGDL-1) + j)

"""Unpack gas DOFs → ω (n×4, with H₂O closure) and p (n), combined GDL then CL."""
function unpack_gas(u::AbstractVector{ElT}, NCL::Int, NGDL::Int) where {ElT<:Real}
    n = NGDL + NCL
    ω = zeros(ElT, n, 4); p = zeros(ElT, n)
    @inbounds for g in 1:n
        base = g <= NGDL ? 5*NCL + 4*(g-1) : 5*NCL + 4*NGDL + 4*(g-NGDL-1)
        w1 = u[base+1]; w2 = u[base+2]; w3 = u[base+3]
        ω[g,1] = w1; ω[g,2] = w2; ω[g,3] = w3; ω[g,4] = max(1 - w1 - w2 - w3, zero(ElT))
        p[g] = u[base+4]
    end
    return ω, p
end

"""
    residual_gas!(F, u, V_cathode, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec; advect, pin_pressure)

Monolithic liquid + gas residual for Stages 3 (advect=false, pin_pressure=true) and
4 (advect=true, pin_pressure=false). φ_s=V_applied_SHE, φ_L=0 (simplified kinetics).
"""
function residual_gas!(F::AbstractVector{ElT}, u::AbstractVector{ElT}, V_cathode::Real,
                       cl_mesh, gdl_mesh, gas_mesh, c_eq_vec;
                       advect::Bool, pin_pressure::Bool) where {ElT<:Real}
    NCL = cl_mesh.N; NGDL = gdl_mesh.N; n = gas_mesh.n
    c = unpack_log_conc(u, NCL)
    ω, p = unpack_gas(u, NCL, NGDL)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # ── kinetics + phase transfer per CL cell (uses local CL gas composition)
    R_CO2R_cell = zeros(ElT, NCL); R_HER_cell = zeros(ElT, NCL); R_PT_cell = zeros(ElT, NCL)
    for ix in 1:NCL
        c_ix = @view c[:, ix]
        R_CO2R_cell[ix] = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1],
                                 a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_HER_cell[ix]  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        g = NGDL + ix
        ωg = (ω[g,1], ω[g,2], ω[g,3], ω[g,4])
        p_CO2_ix = omega_to_x(ωg)[1] * p[g] / 101325.0            # atm
        H_ix = sechenov_H_CO2(c_ix)
        R_PT_cell[ix] = a_v_GL * k_GL() * (H_ix * p_CO2_ix - c_ix[5])   # §7.4
    end

    # ── LIQUID residual: diffusion + Sherwood right BC; CO₂ left BC = Henry(local gas cell 1)
    dxb = cl_mesh.dx_between; dxfl = cl_mesh.dx_face_left; dx = cl_mesh.dx
    g1 = NGDL + 1
    ωg1 = (ω[g1,1], ω[g1,2], ω[g1,3], ω[g1,4])
    p_CO2_1 = omega_to_x(ωg1)[1] * p[g1] / 101325.0
    c_CO2_bc = sechenov_H_CO2(@view c[:,1]) * p_CO2_1
    Jl = zeros(ElT, 5, NCL+1)
    Jl[5,1] = diffusion_flux(c_CO2_bc, c[5,1], D_eff[5], dxfl)
    for f in 2:NCL, i in 1:5
        Jl[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], dxb[f-1])
    end
    for i in 1:5
        Jl[i,NCL+1] = -k_MT[i] * (c_eq_vec[i] - c[i,NCL])
    end
    for ix in 1:NCL
        buf = buffer_sources(@view c[:,ix])
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*R_CO2R_cell[ix] + 2*R_HER_cell[ix]; end
            if i == 5; src -= R_CO2R_cell[ix]; src += R_PT_cell[ix]; end
            F[liquid_idx(ix,i)] = Jl[i,ix] - Jl[i,ix+1] + src * dx[ix]
        end
    end

    # ── GAS residual: SM fluxes at faces 1..n+1 (face1=inlet Dirichlet ghost, faceN+1=zero flux)
    Jg = zeros(ElT, 3, n+1); ρu = zeros(ElT, n+1)
    # inlet face
    let κf = kappa_cell(1, NGDL), bf = gas_mesh.brugg[1],
        ωR = (ω[1,1], ω[1,2], ω[1,3], ω[1,4])
        J1,J2,J3,m = sm_face_flux(omega_inlet_tuple, p_total, ωR, p[1],
                                  gas_mesh.dx_face_left, bf, κf, advect)
        Jg[1,1]=J1; Jg[2,1]=J2; Jg[3,1]=J3; ρu[1]=m
    end
    for f in 2:n
        gl = f-1; gr = f
        bf = harm(gas_mesh.brugg[gl], gas_mesh.brugg[gr])
        κf = harm(kappa_cell(gl,NGDL), kappa_cell(gr,NGDL))
        ωL = (ω[gl,1], ω[gl,2], ω[gl,3], ω[gl,4]); ωR = (ω[gr,1], ω[gr,2], ω[gr,3], ω[gr,4])
        J1,J2,J3,m = sm_face_flux(ωL, p[gl], ωR, p[gr], gas_mesh.dx_between[f-1], bf, κf, advect)
        Jg[1,f]=J1; Jg[2,f]=J2; Jg[3,f]=J3; ρu[f]=m
    end
    # outlet face n+1: zero flux (gas doesn't enter electrolyte) — Jg[:,n+1]=0, ρu[n+1]=0

    for g in 1:n
        dxg = gas_mesh.dx[g]
        if g <= NGDL
            S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT); Smass = zero(ElT)
        else
            ix = g - NGDL
            S1 = -R_PT_cell[ix] * M_gas[1]      # CO₂ lost to liquid
            S2 =  R_CO2R_cell[ix] * M_gas[2]     # CO produced
            S3 =  R_HER_cell[ix]  * M_gas[3]     # H₂ produced
            Smass = S1 + S2 + S3
        end
        F[gas_idx_g(g,1,NCL,NGDL)] = Jg[1,g] - Jg[1,g+1] + S1*dxg
        F[gas_idx_g(g,2,NCL,NGDL)] = Jg[2,g] - Jg[2,g+1] + S2*dxg
        F[gas_idx_g(g,3,NCL,NGDL)] = Jg[3,g] - Jg[3,g+1] + S3*dxg
        if pin_pressure
            F[gas_idx_g(g,4,NCL,NGDL)] = p[g] - p_total              # Stage 3: p pinned
        else
            F[gas_idx_g(g,4,NCL,NGDL)] = ρu[g] - ρu[g+1] + Smass*dxg  # Stage 4: Darcy continuity
        end
    end
    return F
end

residual_stage3!(F, u, V, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec) =
    residual_gas!(F, u, V, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec; advect=false, pin_pressure=true)

residual_stage4!(F, u, V, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec) =
    residual_gas!(F, u, V, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec; advect=true, pin_pressure=false)

# Build a Stage-3/4 warm-start: liquid from a Stage-2 solution, gas at inlet ω / p_total.
function make_u0_gas(u_liquid::Vector{Float64}, NCL::Int, NGDL::Int)
    length(u_liquid) == 5*NCL || error("warm-start DOF mismatch: got $(length(u_liquid)), " *
        "expected $(5*NCL) for N_CL=$NCL — likely a dev/production cache mixup (delete stale cache)")
    n = NGDL + NCL
    u0 = zeros(5*NCL + 4*n)
    u0[1:5*NCL] .= u_liquid[1:5*NCL]
    for g in 1:n
        base = g <= NGDL ? 5*NCL + 4*(g-1) : 5*NCL + 4*NGDL + 4*(g-NGDL-1)
        u0[base+1] = omega_inlet[1]; u0[base+2] = omega_inlet[2]
        u0[base+3] = omega_inlet[3]; u0[base+4] = p_total
    end
    return u0
end

# Darcy diagnostics (Stage 4): per-cell pressure drop & velocity, plus inlet mass-balance check.
function darcy_diagnostics(u, cl_mesh, gdl_mesh, gas_mesh, V)
    NCL = cl_mesh.N; NGDL = gdl_mesh.N; n = gas_mesh.n
    c = unpack_log_conc(u, NCL); ω, p = unpack_gas(u, NCL, NGDL)
    Vapp = V - V_RHE_to_SHE_offset
    Smass_tot = 0.0
    for ix in 1:NCL
        c_ix = @view c[:,ix]
        rco2r = R_CO2R(c_ix[5], Vapp, 0.0, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        rher  = R_HER(Vapp, 0.0, c_ix[1], a_v_CL, n_e)
        g = NGDL + ix; ωg = (ω[g,1],ω[g,2],ω[g,3],ω[g,4])
        p_CO2 = omega_to_x(ωg)[1]*p[g]/101325; H = sechenov_H_CO2(c_ix)
        rpt = a_v_GL*k_GL()*(H*p_CO2 - c_ix[5])
        Smass_tot += (-rpt*M_gas[1] + rco2r*M_gas[2] + rher*M_gas[3]) * cl_mesh.dx[ix]
    end
    uG_face = zeros(n+1)
    uG_face[1] = -(kappa_cell(1,NGDL)/mu_gas)*(p[1]-p_total)/gas_mesh.dx_face_left
    for f in 2:n
        κf = harm(kappa_cell(f-1,NGDL), kappa_cell(f,NGDL))
        uG_face[f] = -(κf/mu_gas)*(p[f]-p[f-1])/gas_mesh.dx_between[f-1]
    end
    uG_cell = [0.5*(uG_face[g]+uG_face[g+1]) for g in 1:n]
    dp_cell = [p[g]-p_total for g in 1:n]
    ρ_in = 0.5*(p_total*Mn_from_omega(omega_inlet_tuple)/(R_gas*T) +
                p[1]*Mn_from_omega((ω[1,1],ω[1,2],ω[1,3],ω[1,4]))/(R_gas*T))
    ρu_in = ρ_in * uG_face[1]
    massbal_relerr = abs(ρu_in - (-Smass_tot)) / max(abs(Smass_tot), 1e-30)
    return dp_cell, uG_cell, ρu_in, Smass_tot, massbal_relerr
end

# Gas profile extraction for plots: returns partial pressures (atm) and p_total (atm) per cell.
function gas_profiles(u, NCL::Int, NGDL::Int)
    ω, p = unpack_gas(u, NCL, NGDL)
    n = NGDL + NCL
    pCO2 = zeros(n); pCO = zeros(n); pH2 = zeros(n); ptot = zeros(n)
    for g in 1:n
        x = omega_to_x((ω[g,1], ω[g,2], ω[g,3], ω[g,4]))
        pCO2[g] = x[1]*p[g]/101325; pCO[g] = x[2]*p[g]/101325
        pH2[g]  = x[3]*p[g]/101325; ptot[g] = p[g]/101325
    end
    return pCO2, pCO, pH2, ptot
end
