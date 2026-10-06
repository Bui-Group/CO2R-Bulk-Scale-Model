# assembly.jl — DOF layout, residuals for every stage, current extractors,
#                and the per-stage binary cache (§2, §3, §7, §10, §11, §17.4)

using Printf

# ──────────────────────────────────────────────────────────────────────────
# DOF layout (§2, §11 Stage 5)
#   u = [ ln c (5·N_CL) | gas (4·Ng) | φ_s (Ng) | φ_L (N_CL) ]
#   with Ng = N_GDL + N_CL, gas/solid cells ordered GDL first then CL.
# ──────────────────────────────────────────────────────────────────────────
struct Layout
    N_CL::Int
    N_GDL::Int
    Ng::Int
    stage::Int
    n_gas::Int
    N_TOT::Int
end

function Layout(N_CL::Int, N_GDL::Int, stage::Int)
    Ng = N_GDL + N_CL
    n_liq = 5 * N_CL
    if stage <= 2
        return Layout(N_CL, N_GDL, Ng, stage, 0, n_liq)
    elseif stage <= 4
        return Layout(N_CL, N_GDL, Ng, stage, 4 * Ng, n_liq + 4 * Ng)
    else
        return Layout(N_CL, N_GDL, Ng, stage, 4 * Ng, n_liq + 4 * Ng + Ng + N_CL)
    end
end

@inline liquid_idx(ix::Int, i::Int) = 5 * (ix - 1) + i
@inline gas_idx(ly::Layout, g::Int, j::Int)  = 5 * ly.N_CL + 4 * (g - 1) + j
@inline phi_s_idx(ly::Layout, g::Int)  = 5 * ly.N_CL + 4 * ly.Ng + g
@inline phi_L_idx(ly::Layout, ix::Int) = 5 * ly.N_CL + 4 * ly.Ng + ly.Ng + ix
@inline gas_cl_cell(ly::Layout, ix::Int) = ly.N_GDL + ix

"Weight blocks for the affine-covariant kernel (§9.5.3 step 1)."
function weight_blocks(ly::Layout)
    b = UnitRange{Int}[1:(5*ly.N_CL)]
    if ly.n_gas > 0
        g0 = 5 * ly.N_CL
        push!(b, (g0 + 1):(g0 + 4*ly.N_GDL))                       # gas GDL
        push!(b, (g0 + 4*ly.N_GDL + 1):(g0 + 4*ly.Ng))             # gas CL
    end
    if ly.stage >= 5
        s0 = 5 * ly.N_CL + 4 * ly.Ng
        push!(b, (s0 + 1):(s0 + ly.N_GDL))                          # φ_s GDL
        push!(b, (s0 + ly.N_GDL + 1):(s0 + ly.Ng))                  # φ_s CL
        push!(b, (s0 + ly.Ng + 1):(s0 + ly.Ng + ly.N_CL))           # φ_L
    end
    return b
end

# ── unpacking helpers (element type follows u, for ForwardDiff) ────────────
function unpack_log_conc(u::AbstractVector{ElT}, N::Int) where {ElT<:Real}
    c = zeros(ElT, 5, N)
    @inbounds for ix in 1:N, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

@inline function cell_gas(u::AbstractVector{ElT}, ly::Layout, g::Int) where {ElT<:Real}
    b = 5 * ly.N_CL + 4 * (g - 1)
    @inbounds w1 = u[b+1]; @inbounds w2 = u[b+2]
    @inbounds w3 = u[b+3]; @inbounds p  = u[b+4]
    w4 = omega_H2O_closure(w1, w2, w3)
    return (w1, w2, w3, w4), p
end

@inline inlet_gas(::Type{ElT}) where {ElT<:Real} =
    (ntuple(j -> convert(ElT, omega_inlet[j]), Val(4)), convert(ElT, p_total))

# ──────────────────────────────────────────────────────────────────────────
# Stages 1 & 2 — liquid only, simplified kinetics (φ_L = 0, φ_s = V_applied)
# ──────────────────────────────────────────────────────────────────────────
function residual_stage12!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                           V_cathode::Real, cl::Mesh1D,
                           c_eq::Vector{Float64},
                           dirichlet_right::Bool) where {ElT<:Real}
    N = cl.N
    V_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N)
    c_CO2_bc = H_vanthoff() * p_CO2_inlet_atm         # §7.1, ≈ 16.0 mol/m³

    J = zeros(ElT, 5, N + 1)
    J[5, 1] = diffusion_flux(convert(ElT, c_CO2_bc), c[5, 1], D_eff[5], cl.dxc[1])
    @inbounds for f in 2:N, i in 1:5
        J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl.dxc[f])
    end
    @inbounds for i in 1:5
        J[i, N+1] = dirichlet_right ?
            diffusion_flux(c[i, N], convert(ElT, c_eq[i]), D_eff[i], cl.dxc[N+1]) :
            -k_MT[i] * (c_eq[i] - c[i, N])            # §7.2 Sherwood Neumann
    end

    @inbounds for ix in 1:N
        c_ix = @view c[:, ix]
        r_co2r = R_CO2R(c_ix[5], V_SHE, zero(ElT), c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*r_co2r + 2*r_her)
            i == 5 && (src -= r_co2r)
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl.dx[ix]
        end
    end
    return F
end

# ──────────────────────────────────────────────────────────────────────────
# Stages 3 & 4 — liquid + Stefan-Maxwell gas (+ Darcy in Stage 4)
# ──────────────────────────────────────────────────────────────────────────
function residual_stage34!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                           V_cathode::Real, cl::Mesh1D, gm::GasMesh,
                           c_eq::Vector{Float64}, ly::Layout,
                           darcy::Bool) where {ElT<:Real}
    N   = cl.N
    Ng  = gm.N
    V_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N)

    # ── faradaic + phase-transfer rates per CL cell
    Rco = zeros(ElT, N); Rh2 = zeros(ElT, N); Rpt = zeros(ElT, N)
    @inbounds for ix in 1:N
        c_ix = @view c[:, ix]
        Rco[ix] = R_CO2R(c_ix[5], V_SHE, zero(ElT), c_ix[1],
                         a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rh2[ix] = R_HER(V_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        wg, pg  = cell_gas(u, ly, gas_cl_cell(ly, ix))
        Rpt[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(c_ix) * pCO2_atm(wg, pg) - c_ix[5])
    end

    # ── liquid fluxes: CO₂(aq) left BC from the local gas phase (§7.3)
    w1, p1 = cell_gas(u, ly, gas_cl_cell(ly, 1))
    c_CO2_bc = sechenov_H_CO2(@view c[:, 1]) * pCO2_atm(w1, p1)

    J = zeros(ElT, 5, N + 1)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], cl.dxc[1])
    @inbounds for f in 2:N, i in 1:5
        J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl.dxc[f])
    end
    @inbounds for i in 1:5
        J[i, N+1] = -k_MT[i] * (c_eq[i] - c[i, N])
    end

    @inbounds for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*Rco[ix] + 2*Rh2[ix])
            i == 5 && (src += -Rco[ix] + Rpt[ix])
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl.dx[ix]
        end
    end

    gas_residual!(F, u, ly, gm, Rco, Rh2, Rpt, darcy)
    return F
end

"""
    gas_residual!(F, u, ly, gm, Rco, Rh2, Rpt, darcy)

Stefan-Maxwell species rows + the pressure row (§6.4).  With `darcy = false`
(Stage 3) u_G ≡ 0 and the pressure row is pinned to `p_total` (§11 Stage 3).
"""
function gas_residual!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                       ly::Layout, gm::GasMesh,
                       Rco::AbstractVector{ElT}, Rh2::AbstractVector{ElT},
                       Rpt::AbstractVector{ElT}, darcy::Bool) where {ElT<:Real}
    Ng = gm.N
    Jg   = zeros(ElT, 4, Ng + 1)
    rhou = zeros(ElT, Ng + 1)

    # face 1 — flow-field Dirichlet ghost (ω_inlet, p_total)
    w_in, p_in = inlet_gas(ElT)
    w1, p1 = cell_gas(u, ly, 1)
    Jf, ru = gas_face_flux(w_in, p_in, w1, p1, D_bin_GDL, kappa_GDL, gm.dxc[1], darcy)
    @inbounds for j in 1:4; Jg[j, 1] = Jf[j]; end
    rhou[1] = ru

    @inbounds for f in 2:Ng
        wL, pL = cell_gas(u, ly, f - 1)
        wR, pR = cell_gas(u, ly, f)
        Jf, ru = gas_face_flux(wL, pL, wR, pR, face_Dbin(gm, f), face_kappa(gm, f),
                               gm.dxc[f], darcy)
        for j in 1:4; Jg[j, f] = Jf[j]; end
        rhou[f] = ru
    end
    # face Ng+1 — electrolyte side: zero gas flux, dp/dx = 0 (already zeros)

    @inbounds for g in 1:Ng
        is_cl = g > gm.N_GDL
        ix = g - gm.N_GDL
        S1 = is_cl ? -Rpt[ix] * M_gas[1] : zero(ElT)
        S2 = is_cl ?  Rco[ix] * M_gas[2] : zero(ElT)
        S3 = is_cl ?  Rh2[ix] * M_gas[3] : zero(ElT)
        dxg = gm.dx[g]
        F[gas_idx(ly, g, 1)] = Jg[1, g] - Jg[1, g+1] + S1 * dxg
        F[gas_idx(ly, g, 2)] = Jg[2, g] - Jg[2, g+1] + S2 * dxg
        F[gas_idx(ly, g, 3)] = Jg[3, g] - Jg[3, g+1] + S3 * dxg
        if darcy
            F[gas_idx(ly, g, 4)] = rhou[g] - rhou[g+1] + (S1 + S2 + S3) * dxg
        else
            F[gas_idx(ly, g, 4)] = u[gas_idx(ly, g, 4)] - p_total
        end
    end
    return F
end

# ──────────────────────────────────────────────────────────────────────────
# Stage 5 — full electrochemistry: NP migration + φ_L DOF + φ_s DOF
# ──────────────────────────────────────────────────────────────────────────
function residual_stage5!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, cl::Mesh1D, gm::GasMesh,
                          c_eq::Vector{Float64}, ly::Layout) where {ElT<:Real}
    N  = cl.N
    Ng = gm.N
    V_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N)

    phi_L = zeros(ElT, N)
    @inbounds for ix in 1:N; phi_L[ix] = u[phi_L_idx(ly, ix)]; end
    phi_s = zeros(ElT, Ng)
    @inbounds for g in 1:Ng;  phi_s[g] = u[phi_s_idx(ly, g)]; end

    # ── rates (local φ_s, φ_L — NOT V_applied)
    Rco = zeros(ElT, N); Rh2 = zeros(ElT, N); Rpt = zeros(ElT, N)
    @inbounds for ix in 1:N
        c_ix = @view c[:, ix]
        ps = phi_s[gm.N_GDL + ix]
        pl = phi_L[ix]
        Rco[ix] = R_CO2R(c_ix[5], ps, pl, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rh2[ix] = R_HER(ps, pl, c_ix[1], a_v_CL, n_e)
        wg, pg  = cell_gas(u, ly, gas_cl_cell(ly, ix))
        Rpt[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(c_ix) * pCO2_atm(wg, pg) - c_ix[5])
    end

    # ── liquid Nernst-Planck fluxes
    w1, p1 = cell_gas(u, ly, gas_cl_cell(ly, 1))
    c_CO2_bc = sechenov_H_CO2(@view c[:, 1]) * pCO2_atm(w1, p1)

    J = zeros(ElT, 5, N + 1)
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    J[5, 1] = zero(ElT)   # was: diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], cl.dxc[1])
    @inbounds for f in 2:N, i in 1:5
        J[i, f] = np_flux(c[i, f-1], c[i, f], phi_L[f-1], phi_L[f],
                          D_eff[i], z_ion[i], cl.dxc[f])
    end
    @inbounds for i in 1:5                       # §11 gotcha #12: migration at the right BC
        L_eff = D_eff[i] / k_MT[i]
        J[i, N+1] = np_flux(c[i, N], convert(ElT, c_eq[i]),
                            phi_L[N], zero(ElT), D_eff[i], z_ion[i], L_eff)
    end

    # ── Cs⁺ NP flux (electroneutrality species; no-flux at both CL faces)
    cCs = zeros(ElT, N)
    @inbounds for ix in 1:N; cCs[ix] = c_Cs_from_electroneutrality(@view c[:, ix]); end
    Jcs = zeros(ElT, N + 1)
    @inbounds for f in 2:N
        Jcs[f] = np_flux(cCs[f-1], cCs[f], phi_L[f-1], phi_L[f],
                         D_eff_Cs, z_Cs, cl.dxc[f])
    end

    # ── liquid species residuals
    @inbounds for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*Rco[ix] + 2*Rh2[ix])
            i == 5 && (src += -Rco[ix] + Rpt[ix])
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl.dx[ix]
        end
    end

    # ── gas residual
    gas_residual!(F, u, ly, gm, Rco, Rh2, Rpt, true)

    # ── φ_L residual: local ionic charge balance (§3.1)
    iL = zeros(ElT, N + 1)
    @inbounds for f in 1:N+1
        s = zero(ElT)
        for i in 1:5; s += z_ion[i] * J[i, f]; end
        iL[f] = F_const * (s + z_Cs * Jcs[f])
    end
    @inbounds for ix in 1:N
        i_far_vol = n_e * F_const * (Rco[ix] + Rh2[ix])
        F[phi_L_idx(ly, ix)] = (iL[ix+1] - iL[ix]) + i_far_vol * cl.dx[ix]
    end
    F[phi_L_idx(ly, N)] = phi_L[N]              # Dirichlet gauge override

    # ── φ_s residual on GDL + CL (§3.4), ghost-cell Dirichlet at the collector
    i_s = zeros(ElT, Ng + 1)
    i_s[1] = -sigma_s_eff_GDL * (phi_s[1] - V_SHE) / gm.dxc[1]
    @inbounds for f in 2:Ng
        sig = f <= gm.N_GDL ? sigma_s_eff_GDL :
              (f == gm.i_int ? sigma_s_eff_INT : sigma_s_eff_CL)
        i_s[f] = -sig * (phi_s[f] - phi_s[f-1]) / gm.dxc[f]
    end
    # i_s[Ng+1] = 0  (Neumann at the electrolyte side)

    @inbounds for g in 1:gm.N_GDL
        F[phi_s_idx(ly, g)] = i_s[g+1] - i_s[g]
    end
    @inbounds for ix in 1:N
        g = gm.N_GDL + ix
        i_far_vol = n_e * F_const * (Rco[ix] + Rh2[ix])
        F[phi_s_idx(ly, g)] = (i_s[g+1] - i_s[g]) - i_far_vol * cl.dx[ix]
    end
    return F
end

# ──────────────────────────────────────────────────────────────────────────
# Current extractors (§17.4) and Stage-5 diagnostics
# ──────────────────────────────────────────────────────────────────────────
"Partial current densities [A/m²] for Stages 1–4 (φ_s = V_applied_SHE, φ_L = 0)."
function compute_currents(u::Vector{Float64}, V_SHE::Float64, cl::Mesh1D)
    c = unpack_log_conc(u, cl.N)
    iCO = 0.0; iH2 = 0.0
    for ix in 1:cl.N
        r1 = R_CO2R(c[5,ix], V_SHE, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r2 = R_HER(V_SHE, 0.0, c[1,ix], a_v_CL, n_e)
        iCO += n_e * F_const * r1 * cl.dx[ix]
        iH2 += n_e * F_const * r2 * cl.dx[ix]
    end
    return (iCO, iH2)
end

"Partial current densities [A/m²] for Stage 5 (local φ_s, φ_L from the DOFs)."
function compute_currents5(u::Vector{Float64}, ly::Layout, cl::Mesh1D)
    c = unpack_log_conc(u, cl.N)
    iCO = 0.0; iH2 = 0.0
    for ix in 1:cl.N
        ps = u[phi_s_idx(ly, ly.N_GDL + ix)]
        pl = u[phi_L_idx(ly, ix)]
        r1 = R_CO2R(c[5,ix], ps, pl, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r2 = R_HER(ps, pl, c[1,ix], a_v_CL, n_e)
        iCO += n_e * F_const * r1 * cl.dx[ix]
        iH2 += n_e * F_const * r2 * cl.dx[ix]
    end
    return (iCO, iH2)
end

"Volumetric rate profiles [mol/m³/s] (stage-aware)."
function rate_profiles(u::Vector{Float64}, ly::Layout, cl::Mesh1D, V_SHE::Float64)
    c = unpack_log_conc(u, cl.N)
    r1 = zeros(cl.N); r2 = zeros(cl.N)
    for ix in 1:cl.N
        ps = ly.stage >= 5 ? u[phi_s_idx(ly, ly.N_GDL + ix)] : V_SHE
        pl = ly.stage >= 5 ? u[phi_L_idx(ly, ix)] : 0.0
        r1[ix] = R_CO2R(c[5,ix], ps, pl, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r2[ix] = R_HER(ps, pl, c[1,ix], a_v_CL, n_e)
    end
    return r1, r2
end

# ──────────────────────────────────────────────────────────────────────────
# §10 Per-stage binary cache
# ──────────────────────────────────────────────────────────────────────────
function save_solution(dir::String, prefix::String, u::Vector{Float64}, V::Float64)
    mkpath(dir)
    fn = joinpath(dir, @sprintf("%sV%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

function load_solution(dir::String, prefix::String, V::Float64)
    fn = joinpath(dir, @sprintf("%sV%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

function list_cache(dir::String, prefix::String)
    isdir(dir) || return Float64[]
    files = filter(f -> startswith(f, prefix) && endswith(f, ".bin"), readdir(dir))
    isempty(files) && return Float64[]
    return sort([parse(Float64, f[length(prefix)+2:end-4]) for f in files])
end
