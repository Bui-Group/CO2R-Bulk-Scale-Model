# assembly.jl — residual assembly + current extractors (Stages 1+; grows per stage)
#
# Liquid DOF layout (all stages): u[5*(ix-1)+i] = ln(c_i) at CL cell ix,
# species order [H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq)].

@inline liquid_idx(ix, i) = 5 * (ix - 1) + i

# Unpack the liquid log-concentration block → 5×N matrix, eltype-matched (ForwardDiff).
function unpack_log_conc(u::AbstractVector{ElT}, N::Int) where {ElT<:Real}
    c = zeros(ElT, 5, N)
    @inbounds for ix in 1:N, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

# ── Stage 1 residual: liquid NP diffusion only + Dirichlet both sides + simplified
#    kinetics (φ_s = V_applied_SHE, φ_L = 0). §11 Stage 1.
function residual_stage1!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, m::Mesh,
                          c_CO2_bc::Float64, c_eq::Vector{Float64}) where {ElT<:Real}
    N = m.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # diffusion fluxes, J[i, face], faces 1..N+1
    J = zeros(ElT, 5, N + 1)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], m.dx_cc[1])   # CO₂ Dirichlet (GDL)
    # J[1:4,1] = 0 (ion no-flux at GDL) — already zero
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], m.dx_cc[f])
        end
    end
    for i in 1:5
        J[i, N+1] = diffusion_flux(c[i, N], c_eq[i], D_eff[i], m.dx_cc[N+1])   # Dirichlet (electrolyte)
    end

    for ix in 1:N
        cix = @view c[:, ix]
        r_co2r = R_CO2R(cix[5], V_applied_SHE, 0.0, cix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, 0.0, cix[1], a_v_CL, n_e)
        buf = buffer_sources(cix)
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*r_co2r + 2*r_her)     # OH⁻ produced by both reactions
            i == 5 && (src -= r_co2r)                 # CO₂(aq) consumed by CO₂R
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * m.dx[ix]
        end
    end
    return F
end

# ── Stage 2 residual: Stage 1 with the electrolyte-side Dirichlet replaced by a
#    Sherwood-Reynolds convective Neumann BC (§7.2). Only the right face changes.
function residual_stage2!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, m::Mesh,
                          c_CO2_bc::Float64, c_eq::Vector{Float64}) where {ElT<:Real}
    N = m.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    J = zeros(ElT, 5, N + 1)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], m.dx_cc[1])
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], m.dx_cc[f])
        end
    end
    # electrolyte-side convective Neumann: J = −k_MT (c_eq − c_N) = +k_MT (c_N − c_eq)
    for i in 1:5
        J[i, N+1] = -k_MT[i] * (c_eq[i] - c[i, N])
    end

    for ix in 1:N
        cix = @view c[:, ix]
        r_co2r = R_CO2R(cix[5], V_applied_SHE, 0.0, cix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, 0.0, cix[1], a_v_CL, n_e)
        buf = buffer_sources(cix)
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*r_co2r + 2*r_her)
            i == 5 && (src -= r_co2r)
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * m.dx[ix]
        end
    end
    return F
end

# ── Stage 3 residual: Stage 2 liquid + Stefan-Maxwell gas (u_G=0) + R_PT phase
#    transfer. Pressure PINNED to p_total (Darcy finalized in Stage 4). §7.3/§7.4/§6.
#    DOFs: 5·N_CL liquid + 4·(N_GDL+N_CL) gas. Simplified kinetics (φ_s=V_applied, φ_L=0).
#    `advect` toggles the Darcy convective term + pressure equation (Stage 4).
function residual_stage3!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, clm::Mesh, gm::GasMesh,
                          c_eq::Vector{Float64}; advect::Bool = false) where {ElT<:Real}
    N_CL = clm.N; N_GDL = gm.N_GDL; Ng = gm.Ng
    c = unpack_log_conc(u, N_CL)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # ── kinetics + phase transfer per CL cell (gas CL cell g = N_GDL + ix) ──
    Rco2r = zeros(ElT, N_CL); Rher = zeros(ElT, N_CL); Rpt = zeros(ElT, N_CL)
    for ix in 1:N_CL
        cix = @view c[:, ix]
        Rco2r[ix] = R_CO2R(cix[5], V_applied_SHE, 0.0, cix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rher[ix]  = R_HER(V_applied_SHE, 0.0, cix[1], a_v_CL, n_e)
        p_co2 = pCO2_atm(u, N_CL, N_GDL + ix)
        H_loc = sechenov_H_CO2(cix)
        Rpt[ix] = a_v_GL * k_GL() * (H_loc * p_co2 - cix[5])
    end

    # ── Darcy velocity per gas face (Stage 4); u_G=0 in Stage 3 ──
    uG = zeros(ElT, Ng + 1)
    if advect
        for f in 2:Ng
            pL = gas_p(u, N_CL, f-1); pR = gas_p(u, N_CL, f)
            uG[f] = -(gm.kappa_face[f] / mu_gas) * (pR - pL) / gm.dx_cc[f]
        end
        # inlet face 1: pressure Dirichlet p_total on the left
        pR1 = gas_p(u, N_CL, 1)
        uG[1] = -(gm.kappa_face[1] / mu_gas) * (pR1 - p_total) / gm.dx_cc[1]
        # right boundary face Ng+1: dp/dx = 0 ⇒ uG = 0
    end

    # ── liquid residual (Stage 2 + CO₂ left BC from local gas + R_PT source) ──
    Jl = zeros(ElT, 5, N_CL + 1)
    p_co2_face = pCO2_atm(u, N_CL, N_GDL + 1)                 # first CL gas cell
    cCO2_bc = sechenov_H_CO2(@view c[:, 1]) * p_co2_face      # Henry-eq liquid CO₂ at x=0
    Jl[5, 1] = diffusion_flux(cCO2_bc, c[5, 1], D_eff[5], clm.dx_cc[1])
    for f in 2:N_CL, i in 1:5
        Jl[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], clm.dx_cc[f])
    end
    for i in 1:5
        Jl[i, N_CL+1] = -k_MT[i] * (c_eq[i] - c[i, N_CL])     # Sherwood
    end
    for ix in 1:N_CL
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*Rco2r[ix] + 2*Rher[ix])
            i == 5 && (src -= Rco2r[ix]; src += Rpt[ix])
            F[liquid_idx(ix, i)] = Jl[i, ix] - Jl[i, ix+1] + src * clm.dx[ix]
        end
    end

    # ── gas residual (Stefan-Maxwell, integrated FV) ──
    Jg = zeros(ElT, 3, Ng + 1)
    # inlet face 1: Dirichlet ω_inlet at p_total (ghost left neighbor)
    let JJ = sm_flux(omega_inlet_tuple, gas_omega(u, N_CL, 1), p_total, gas_p(u, N_CL, 1),
                     gm.brugg[1], gm.dx_cc[1], uG[1])
        Jg[1,1] = JJ[1]; Jg[2,1] = JJ[2]; Jg[3,1] = JJ[3]
    end
    for f in 2:Ng
        JJ = sm_flux(gas_omega(u, N_CL, f-1), gas_omega(u, N_CL, f),
                     gas_p(u, N_CL, f-1), gas_p(u, N_CL, f), gm.brugg[f], gm.dx_cc[f], uG[f])
        Jg[1,f] = JJ[1]; Jg[2,f] = JJ[2]; Jg[3,f] = JJ[3]
    end
    # right boundary face Ng+1: zero flux (already zero)

    for g in 1:Ng
        S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT)
        if g > N_GDL
            ix = g - N_GDL
            S1 = -Rpt[ix]   * M_gas[1]     # CO₂ lost to liquid
            S2 =  Rco2r[ix] * M_gas[2]     # CO produced
            S3 =  Rher[ix]  * M_gas[3]     # H₂ produced
        end
        F[gidx(N_CL, g, 1)] = Jg[1, g] - Jg[1, g+1] + S1 * gm.dx[g]
        F[gidx(N_CL, g, 2)] = Jg[2, g] - Jg[2, g+1] + S2 * gm.dx[g]
        F[gidx(N_CL, g, 3)] = Jg[3, g] - Jg[3, g+1] + S3 * gm.dx[g]
        # pressure equation
        if advect
            # Darcy total-mass continuity: (ρu)_L − (ρu)_R + S_mass·dx = 0
            ρu_L = darcy_massflux(u, N_CL, g, g, gm, uG, true)       # left face g
            ρu_R = darcy_massflux(u, N_CL, g, g+1, gm, uG, false)    # right face g+1
            S_mass = S1 + S2 + S3
            F[gidx(N_CL, g, 4)] = ρu_L - ρu_R + S_mass * gm.dx[g]
        else
            F[gidx(N_CL, g, 4)] = gas_p(u, N_CL, g) - p_total       # Stage 3: pinned
        end
    end
    return F
end

# Darcy mass flux ρ·u_G at a face [kg/m²/s]. `face` is the global face index;
# `g` is the owning cell (used only to pick the upwind/interface density sensibly).
# We use the face-average density from the two adjacent cells (inlet uses p_total, ω of cell 1).
function darcy_massflux(u::AbstractVector{<:Real}, N_CL::Int, g::Int, face::Int,
                        gm::GasMesh, uG::AbstractVector{<:Real}, is_left::Bool)
    Ng = gm.Ng
    if face == 1
        ω = gas_omega(u, N_CL, 1)
        ρ = p_total * Mn_from_omega(ω) / (R_gas * T)     # inlet: p_total, cell-1 composition
        return ρ * uG[1]
    elseif face == Ng + 1
        return zero(eltype(u))                            # no-flow at electrolyte side
    else
        gL = face - 1; gR = face
        ωL = gas_omega(u, N_CL, gL); ωR = gas_omega(u, N_CL, gR)
        ρL = gas_p(u, N_CL, gL) * Mn_from_omega(ωL) / (R_gas * T)
        ρR = gas_p(u, N_CL, gR) * Mn_from_omega(ωR) / (R_gas * T)
        return 0.5 * (ρL + ρR) * uG[face]
    end
end

# ── Current extractors (Stages 1–4 simplified kinetics: pass V_applied_SHE, φ_L=0).
#    Return geometric partial current density in A/m².
function compute_i_CO(u::Vector{Float64}, phi_s::Float64, m::Mesh)
    c = unpack_log_conc(u, m.N)
    i = 0.0
    for ix in 1:m.N
        r = R_CO2R(c[5, ix], phi_s, 0.0, c[1, ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        i += n_e * F_const * r * m.dx[ix]
    end
    return i
end

function compute_i_H2(u::Vector{Float64}, phi_s::Float64, m::Mesh)
    c = unpack_log_conc(u, m.N)
    i = 0.0
    for ix in 1:m.N
        r = R_HER(phi_s, 0.0, c[1, ix], a_v_CL, n_e)
        i += n_e * F_const * r * m.dx[ix]
    end
    return i
end

# Stage 5 current extractors: read local φ_s, φ_L per cell.
function compute_i_CO_local(u::Vector{Float64}, phi_s::Vector{Float64},
                            phi_L::Vector{Float64}, m::Mesh)
    c = unpack_log_conc(u, m.N)
    i = 0.0
    for ix in 1:m.N
        r = R_CO2R(c[5, ix], phi_s[ix], phi_L[ix], c[1, ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        i += n_e * F_const * r * m.dx[ix]
    end
    return i
end
function compute_i_H2_local(u::Vector{Float64}, phi_s::Vector{Float64},
                            phi_L::Vector{Float64}, m::Mesh)
    c = unpack_log_conc(u, m.N)
    i = 0.0
    for ix in 1:m.N
        r = R_HER(phi_s[ix], phi_L[ix], c[1, ix], a_v_CL, n_e)
        i += n_e * F_const * r * m.dx[ix]
    end
    return i
end


# ── §9.5.3 weight blocks for the Stages 1–4 DOF layouts: liquid log-c | gas GDL | gas CL.
#    (Stage 5 appends φ_s GDL | φ_s CL | φ_L — see dof_blocks(::Stage5Layout) in assembly_stage5.jl.)
#    Gas cells g = 1..N_GDL are the GDL, g = N_GDL+1..N_GDL+N_CL the CL (gidx layout above).
function dof_blocks(stage::Int, N_CL::Int, N_GDL::Int)
    blocks = UnitRange{Int}[1:5*N_CL]
    if stage >= 3
        Ng = N_GDL + N_CL
        push!(blocks, gidx(N_CL, 1, 1):gidx(N_CL, N_GDL, 4))         # gas GDL block
        push!(blocks, gidx(N_CL, N_GDL + 1, 1):gidx(N_CL, Ng, 4))    # gas CL block
    end
    return blocks
end

# ── Fixed structural sparsity pattern for Stages 3/4/Fit (§9.3.3, §9.5.2): a guaranteed
#    superset of the FV Jacobian — liquid 5×5 block-tridiagonal; liquid[ix] ↔ CL-gas cell
#    (N_GDL+ix) local coupling both ways (R_PT, faradaic gas sources, CO₂ left BC from CL gas
#    cell 1); gas 4×4 block-tridiagonal over the combined GDL+CL mesh (SM fluxes, Darcy
#    velocity and mass flux all use adjacent cells only). Fixed ⇒ the symbolic LU is reused.
function build_gas_pattern(N_CL::Int, N_GDL::Int)
    rows = Int[]; cols = Int[]
    add!(r, c) = (push!(rows, r); push!(cols, c))
    for ix in 1:N_CL, jx in max(1, ix-1):min(N_CL, ix+1), a in 1:5, b in 1:5
        add!(liquid_idx(ix, a), liquid_idx(jx, b))
    end
    for ix in 1:N_CL, a in 1:5, b in 1:4
        add!(liquid_idx(ix, a), gidx(N_CL, N_GDL + ix, b))
        add!(gidx(N_CL, N_GDL + ix, b), liquid_idx(ix, a))
    end
    Ng = N_GDL + N_CL
    for g in 1:Ng, h in max(1, g-1):min(Ng, g+1), a in 1:4, b in 1:4
        add!(gidx(N_CL, g, a), gidx(N_CL, h, b))
    end
    return rows, cols
end

# ── Stage 3/4 anchor guess (§9.5.5 table): the given liquid block (bulk equilibrium for the
#    self-anchoring sweep) + inlet ω and p_total in every gas cell.
function make_u0_stage3(u_liq::Vector{Float64}, Ng::Int, N_CL::Int)
    Ndof = 5*N_CL + 4*Ng
    u = zeros(Ndof)
    u[1:5*N_CL] .= u_liq[1:5*N_CL]
    for g in 1:Ng
        b = gas_base(N_CL) + 4*(g-1)
        u[b+1] = omega_inlet[1]; u[b+2] = omega_inlet[2]; u[b+3] = omega_inlet[3]
        u[b+4] = p_total
    end
    return u
end

nothing
