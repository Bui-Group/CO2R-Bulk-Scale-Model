# assembly_stage3.jl — Stage 3 monolithic residual (liquid + Stefan-Maxwell gas).
# Adds: gas DOFs (SM diffusion, u_G=0), pinned pressure, volumetric CO₂ phase
# transfer R_PT, gas→liquid Henry left BC, and gas-side faradaic CO/H₂ sources.
# Kinetics still simplified: φ_s = V_applied_SHE, φ_L = 0. (§6, §7.3, §7.4, §Stage 3)

const _W_INLET = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])

function residual_stage3!(F::AbstractVector{<:Real}, u::AbstractVector{<:Real},
                          V_cathode::Real, cl_mesh, gdl_mesh, c_eq::AbstractVector)
    ElT = eltype(u)
    N   = cl_mesh.N
    NG  = gdl_mesh.N
    Ngas = NG + N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # ── Per-CL-cell kinetics + phase transfer ──
    R_co2r = zeros(ElT, N); R_her = zeros(ElT, N); R_pt = zeros(ElT, N)
    for ix in 1:N
        c_ix = @view c[:, ix]
        R_co2r[ix] = R_CO2R(c_ix[5], V_applied_SHE, zero(ElT), c_ix[1],
                            a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_her[ix]  = R_HER(V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        g_cl = NG + ix
        pco2 = pCO2_atm(u, g_cl, N, NG)
        H_loc = sechenov_H(c_ix)
        R_pt[ix] = a_v_GL * k_GL() * (H_loc * pco2 - c_ix[5])     # R_PT > 0 ⇒ gas→liquid
    end

    # ── Liquid Nernst-Planck DIFFUSION fluxes ──
    Jl = zeros(ElT, 5, N + 1)
    # Left CO₂(aq): Henry equilibrium with local gas CO₂ at the GDL/CL interface (CL gas cell 1)
    pco2_left = pCO2_atm(u, NG + 1, N, NG)
    c_CO2_bc  = sechenov_H(@view c[:, 1]) * pco2_left
    Jl[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], cl_mesh.dx_face_left)
    # Jl[1:4,1] = 0 (ions no-flux at GDL side)
    for f in 2:N, i in 1:5
        Jl[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl_mesh.dx_cc[f-1])
    end
    for i in 1:5     # right: Sherwood-Reynolds convective Neumann (np_flux-consistent sign)
        Jl[i, N+1] = k_MT[i] * (c[i, N] - c_eq[i])
    end

    # ── Liquid species residuals (integrated FV form) ──
    for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*R_co2r[ix] + 2*R_her[ix]; end          # OH⁻ produced
            if i == 5; src += -R_co2r[ix] + R_pt[ix];     end          # CO₂: consumed + R_PT replenish
            F[liquid_idx(ix, i)] = Jl[i, ix] - Jl[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end

    # ── Gas-phase Stefan-Maxwell residuals ──
    # Face fluxes Jg[1:3, 1:Ngas+1]; face f is to the LEFT of gas cell f.
    Jg = zeros(ElT, 3, Ngas + 1)
    # f = 1: flow-field Dirichlet (inlet ω, p_total) → GDL cell 1
    w1, p1 = gas_state(u, 1, N, NG)
    fL = sm_face_flux(_W_INLET, ElT(p_total), w1, p1, bf_GDL, gdl_mesh.dx_face_left)
    Jg[1,1], Jg[2,1], Jg[3,1] = fL[1], fL[2], fL[3]
    # interior + interface faces f = 2 .. Ngas
    for f in 2:Ngas
        wL, pL = gas_state(u, f-1, N, NG)
        wR, pR = gas_state(u, f,   N, NG)
        if f <= NG                       # interior GDL face
            bf = bf_GDL; dxc = gdl_mesh.dx_cc[f-1]
        elseif f == NG + 1               # GDL/CL interface
            bf = 2*bf_GDL*bf_CL/(bf_GDL + bf_CL)              # harmonic mean
            dxc = gdl_mesh.dx_face_right + cl_mesh.dx_face_left
        else                             # interior CL face
            bf = bf_CL; dxc = cl_mesh.dx_cc[f - NG - 1]
        end
        ff = sm_face_flux(wL, pL, wR, pR, bf, dxc)
        Jg[1,f], Jg[2,f], Jg[3,f] = ff[1], ff[2], ff[3]
    end
    # f = Ngas+1: right (electrolyte) boundary — zero gas flux (already zero)

    # Species residual + pinned pressure per gas cell
    for g in 1:Ngas
        dxg = g <= NG ? gdl_mesh.dx[g] : cl_mesh.dx[g - NG]
        # sources
        S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT)
        if g > NG
            ix = g - NG
            S1 = -R_pt[ix]   * M_gas[1]      # CO₂ lost to liquid
            S2 =  R_co2r[ix] * M_gas[2]      # CO produced
            S3 =  R_her[ix]  * M_gas[3]      # H₂ produced
        end
        F[gas_dof(g,1,N,NG)] = Jg[1,g] - Jg[1,g+1] + S1 * dxg
        F[gas_dof(g,2,N,NG)] = Jg[2,g] - Jg[2,g+1] + S2 * dxg
        F[gas_dof(g,3,N,NG)] = Jg[3,g] - Jg[3,g+1] + S3 * dxg
        # Stage 3: pressure pinned to p_total (Darcy finalized in Stage 4)
        _, pg = gas_state(u, g, N, NG)
        F[gas_dof(g,4,N,NG)] = pg - p_total
    end
    return F
end

"DOF count for the Stage 3/4 layout (liquid + gas GDL + gas CL)."
n_dof_stage34(N_CL, N_GDL) = 5*N_CL + 4*(N_GDL + N_CL)

"Build the initial gas DOF block at inlet composition (= the gas-only solution)."
function init_gas_inlet!(u, N_CL, N_GDL)
    for g in 1:(N_GDL + N_CL)
        u[gas_dof(g,1,N_CL,N_GDL)] = omega_inlet[1]
        u[gas_dof(g,2,N_CL,N_GDL)] = omega_inlet[2]
        u[gas_dof(g,3,N_CL,N_GDL)] = omega_inlet[3]
        u[gas_dof(g,4,N_CL,N_GDL)] = p_total
    end
    return u
end

"""
    make_u0_stage34_anchor(N_CL, N_GDL, c_eq)

Anchor guess for the Stage 3/4 continuation (§9.5.5 table): bulk buffer equilibrium in
every CL cell, inlet gas composition and p_total in every gas cell. Built from physics,
not from a previous-stage cache.
"""
function make_u0_stage34_anchor(N_CL, N_GDL, c_eq)
    u = zeros(n_dof_stage34(N_CL, N_GDL))
    lnc = log.(c_eq)
    for ix in 1:N_CL; u[liquid_idx(ix,1):liquid_idx(ix,5)] .= lnc; end
    init_gas_inlet!(u, N_CL, N_GDL)
    return u
end

"""
    build_gas_pattern(N_CL, N_GDL) -> (rows, cols)

Fixed structural sparsity pattern for the Stage 3/4 layout (§9.3.3): liquid 5×5
block-tridiagonal; liquid[ix] ↔ CL-gas[ix] local coupling in both directions (R_PT,
faradaic gas sources, Henry left BC from CL gas cell 1); gas 4×4 block-tridiagonal over
the combined GDL+CL gas mesh. A guaranteed superset of the FV Jacobian sparsity, so the
§9.5.2 symbolic LU analysis is reused for the whole sweep.
"""
function build_gas_pattern(N_CL::Int, N_GDL::Int)
    rows = Int[]; cols = Int[]
    add!(r, c) = (push!(rows, r); push!(cols, c))
    for ix in 1:N_CL, jx in max(1, ix-1):min(N_CL, ix+1), a in 1:5, b in 1:5
        add!(liquid_idx(ix, a), liquid_idx(jx, b))
    end
    for ix in 1:N_CL, a in 1:5, b in 1:4
        add!(liquid_idx(ix, a), gas_cl_idx(ix, b, N_CL, N_GDL))
        add!(gas_cl_idx(ix, b, N_CL, N_GDL), liquid_idx(ix, a))
    end
    NC = N_GDL + N_CL
    for g in 1:NC, h in max(1, g-1):min(NC, g+1), a in 1:4, b in 1:4
        add!(gas_dof(g, a, N_CL, N_GDL), gas_dof(h, b, N_CL, N_GDL))
    end
    return rows, cols
end
