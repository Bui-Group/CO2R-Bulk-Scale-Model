# assembly.jl — DOF layout, Stage 1 residual, current extractors, binary cache.
using Printf

# Liquid DOF index (Stages 1–2): species i at CL cell ix
liquid_idx(ix, i) = 5*(ix - 1) + i

"""Unpack log-concentration DOFs → 5×N physical concentration matrix (type-generic)."""
function unpack_log_conc(u::AbstractVector{ElT}, N::Int) where {ElT<:Real}
    c = zeros(ElT, 5, N)
    @inbounds for ix in 1:N, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

"""
    residual_stage1!(F, u, V_cathode, cl_mesh, c_eq_vec)

Stage 1 residual (§11): liquid NP **diffusion only** + Dirichlet both sides +
simplified kinetics (φ_s = V_applied_SHE, φ_L = 0). Integrated FV form.
"""
function residual_stage1!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, cl_mesh, c_eq_vec) where {ElT<:Real}
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c_CO2_henry = H_vanthoff() * 0.88          # §7.1: H_eff(50°C)·p_CO2_inlet ≈ 16.0 mol/m³

    dxb  = cl_mesh.dx_between
    dxfl = cl_mesh.dx_face_left
    dxfr = cl_mesh.dx_face_right
    dx   = cl_mesh.dx

    # ── Diffusion fluxes (NO migration in Stage 1)
    J = zeros(ElT, 5, N+1)
    J[5, 1] = diffusion_flux(c_CO2_henry, c[5,1], D_eff[5], dxfl)
    # J[1:4,1] = 0 (ions: no-flux at GDL side) — already zero
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], dxb[f-1])
        end
    end
    for i in 1:5
        J[i, N+1] = diffusion_flux(c[i, N], c_eq_vec[i], D_eff[i], dxfr)
    end

    # ── Cell residuals: integrated FV (J_left − J_right + S·dx)
    for ix in 1:N
        c_ix = @view c[:, ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end
            if i == 5; src -= r_co2r; end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end
    return F
end

"""
    residual_stage2!(F, u, V_cathode, cl_mesh, c_eq_vec)

Stage 2 (§7.2, §11): identical to Stage 1 except the electrolyte-side (right) BC is a
Sherwood-Reynolds convective Neumann instead of Dirichlet. Only the last face flux changes:
`J[i,N+1] = -k_MT[i]·(c_eq[i] - c[i,N])` (transcribed verbatim, §11 Stage 2).
"""
function residual_stage2!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, cl_mesh, c_eq_vec) where {ElT<:Real}
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c_CO2_henry = H_vanthoff() * 0.88

    dxb  = cl_mesh.dx_between
    dxfl = cl_mesh.dx_face_left
    dx   = cl_mesh.dx

    J = zeros(ElT, 5, N+1)
    J[5, 1] = diffusion_flux(c_CO2_henry, c[5,1], D_eff[5], dxfl)
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], dxb[f-1])
        end
    end
    # ── Stage 2 change: Sherwood-Reynolds convective Neumann at electrolyte side
    for i in 1:5
        J[i, N+1] = -k_MT[i] * (c_eq_vec[i] - c[i, N])
    end

    for ix in 1:N
        c_ix = @view c[:, ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end
            if i == 5; src -= r_co2r; end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end
    return F
end

# ─────────────────────────────────────────────────────────────────────────────
# Current extractors (§17.4). Stages 1–4: pass V_applied_SHE as φ_s, 0.0 as φ_L.
function compute_i_CO(u, V_applied_SHE, cl_mesh)
    c_all = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        r = R_CO2R(c_all[5,ix], V_applied_SHE, 0.0, c_all[1,ix],
                   a_v_CL, lambda_MHC(), i0_CO2R(), 2)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i                                   # A/m²
end

function compute_i_H2(u, V_applied_SHE, cl_mesh)
    c_all = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        r = R_HER(V_applied_SHE, 0.0, c_all[1,ix], a_v_CL, 2)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i                                   # A/m²
end

# ─────────────────────────────────────────────────────────────────────────────
# Binary cache (§10.3) — raw I/O, DOF count header.
function save_solution(dir::String, prefix::String, u::Vector{Float64}, V::Float64)
    mkpath(dir)
    fn = joinpath(dir, @sprintf("%s_V%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

function load_solution(dir::String, prefix::String, V::Float64)
    fn = joinpath(dir, @sprintf("%s_V%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

function list_cache(dir::String, prefix::String)
    isdir(dir) || return Float64[]
    files = filter(f -> startswith(f, prefix*"_V") && endswith(f, ".bin"), readdir(dir))
    sort([parse(Float64, f[length(prefix)+3:end-4]) for f in files])
end
