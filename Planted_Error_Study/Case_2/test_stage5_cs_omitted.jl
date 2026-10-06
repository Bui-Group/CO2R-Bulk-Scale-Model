# ============================================================================
# test_stage5_cs_omitted.jl — EXPERIMENT, not a production driver.
#
# Re-measures how the §9.5 solver stack behaves on the Stage-5 system as §3.1 of
# Case_2_guide.md literally specifies it: the ionic current i_L summed over the five
# transported species only (no Cs⁺ term) with the cell-N φ_L gauge override in place.
# The production residual in potentials.jl (Cs⁺ restored in i_L, override removed —
# the approved fix) is NOT modified: the literal variant `residual_stage5_literal!`
# is defined locally below and differs from `residual_stage5!` only in its φ_L block.
#
# Prints, for the build dialogue's blocker narrative:
#   A. ‖F‖∞ of the literal residual at the constructed V_c = 0 anchor guess;
#   B. the algebraic identity F_φL[ix] = −F·Σᵢ zᵢ F_species[ix] (cells 1..N−1) at a
#      generic non-equilibrium state, for the literal residual;
#   C. dense-AD-Jacobian SVD, literal vs production residual at the same states (and
#      Stage 4 at a fit-cache state for comparison): σ_max, σ_min, cond, and the count
#      of σ < 1e-10·σ_max (the "near-null dimension");
#   D. the Stage-5 anchor solve (newton_affine!, production anchor settings) on the
#      literal residual: converged / iterations / reason / err_est / θ / α / ‖F‖∞, plus
#      the largest raw liquid log-step before the §9.5.3 clamp on the first iteration;
#   E. the continuation chain (pre-ramp + 89 outputs) on the literal residual, carrying
#      the previous state forward after a failure — the production driver aborts at a
#      failed anchor, this reports the per-step signature regardless (no sub-step
#      recovery);
#   F. newton_affine! from fit-cache warm-starts (Stage-4 fit state + φ_s = V_app,
#      φ_L = 0) at the grid points nearest the five experimental voltages.
#
# Usage:  julia test_stage5_cs_omitted.jl        (after run_fit.jl; several minutes)
#   env CS_TEST_MAX_POINTS=n   truncate the chain in E to its first n points
#   env CS_TEST_SKIP_CHAIN=1   skip E
# ============================================================================

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "gas_transport.jl"))
include(joinpath(HERE, "solver_sparse.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "potentials.jl"))
include(joinpath(HERE, "cache.jl"))

const FIT_DIR   = joinpath(HERE, "output_fit")
const FIT_CACHE = joinpath(FIT_DIR, "cache")
const S5_CACHE  = joinpath(HERE, "output_stage5", "cache")

function load_fitted!()
    fn = joinpath(FIT_DIR, "fitted_params.txt")
    isfile(fn) || error("No fitted_params.txt — run run_fit.jl first (the blocker was hit with fitted kinetics)")
    for ln in readlines(fn)
        isempty(strip(ln)) && continue
        k, v = split(ln, "="); val = parse(Float64, v)
        k == "i0_CO2R"       && set_i0_CO2R!(val)
        k == "lambda_MHC"    && set_lambda_MHC!(val)
        k == "HER_prefactor" && set_HER_prefactor!(val)
        k == "alpha_c_HER"   && set_alpha_c_HER!(val)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# LITERAL §3.1 Stage-5 residual. Identical to potentials.jl residual_stage5!
# except the φ_L block: i_L = F Σ_{5 species} z_i J_i (no Cs⁺ NP current) and the
# §3.1 gauge override F[φ_L(N)] = φ_L[N].
# ─────────────────────────────────────────────────────────────────────────────
function residual_stage5_literal!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                                  V_cathode::Real, cl_mesh::Mesh, gas_mesh::Mesh,
                                  c_eq_vec::AbstractVector{<:Real}, ly::GasLayout) where {ElT<:Real}
    N = cl_mesh.N; Ng = gas_mesh.N; NGDL = ly.N_GDL
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    phi_s = Vector{ElT}(undef, Ng)
    for ig in 1:NGDL; phi_s[ig] = u[phi_s_gdl_idx5(ly, ig)]; end
    for ix in 1:N;    phi_s[NGDL + ix] = u[phi_s_cl_idx5(ly, ix)]; end
    phi_L = ElT[u[phi_L_idx5(ly, ix)] for ix in 1:N]

    R_co2r = zeros(ElT, N); R_her = zeros(ElT, N); R_pt = zeros(ElT, N)
    for ix in 1:N
        cix = @view c[:, ix]
        ps = u[phi_s_cl_idx5(ly, ix)]; pL = phi_L[ix]
        R_co2r[ix] = R_CO2R(cix[5], ps, pL, cix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_her[ix]  = R_HER(ps, pL, cix[1], a_v_CL, n_e)
        pco2 = pCO2_atm(u, ly, NGDL + ix)
        R_pt[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(cix) * pco2 - cix[5])
    end

    dxb = cl_mesh.dx_between; dxfL = cl_mesh.dx_face_left; dx = cl_mesh.dx
    J = zeros(ElT, 5, N+1)
    c_CO2_bc = sechenov_H_CO2(@view c[:, 1]) * pCO2_atm(u, ly, NGDL + 1)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], dxfL)
    for f in 2:N, i in 1:5
        J[i, f] = np_flux(c[i, f-1], c[i, f], phi_L[f-1], phi_L[f], D_eff[i], z_charge[i], dxb[f-1])
    end
    for i in 1:5
        L_eff = D_eff[i] / k_MT[i]
        J[i, N+1] = np_flux(c[i, N], c_eq_vec[i], phi_L[N], zero(ElT), D_eff[i], z_charge[i], L_eff)
    end

    for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*R_co2r[ix] + 2*R_her[ix])
            i == 5 && (src += -R_co2r[ix] + R_pt[ix])
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end

    Jg = zeros(ElT, 4, Ng+1); rhou = zeros(ElT, Ng+1)
    let p1 = gas_cell_p(u, ly, 1)
        uG = -(kappa_GDL / mu_gas) * (p1 - p_total) / gas_mesh.dx_face_left
        Jf, ru = sm_face_flux_conv(gas_inlet_tuple(), gas_cell_omega(u, ly, 1),
                                   0.5*(p_total + p1), eps_GDL^1.5, gas_mesh.dx_face_left, uG)
        for j in 1:4; Jg[j, 1] = Jf[j]; end
        rhou[1] = ru
    end
    for f in 2:Ng
        bf = harmonic_mean(gas_cell_eps(ly, f-1)^1.5, gas_cell_eps(ly, f)^1.5)
        κf = harmonic_mean(gas_cell_kappa(ly, f-1), gas_cell_kappa(ly, f))
        pL = gas_cell_p(u, ly, f-1); pR = gas_cell_p(u, ly, f)
        uG = -(κf / mu_gas) * (pR - pL) / gas_mesh.dx_between[f-1]
        Jf, ru = sm_face_flux_conv(gas_cell_omega(u, ly, f-1), gas_cell_omega(u, ly, f),
                                   0.5*(pL + pR), bf, gas_mesh.dx_between[f-1], uG)
        for j in 1:4; Jg[j, f] = Jf[j]; end
        rhou[f] = ru
    end
    dxg = gas_mesh.dx
    for cc in 1:Ng
        SCO2 = zero(ElT); SCO = zero(ElT); SH2 = zero(ElT)
        if cc > NGDL
            ix = cc - NGDL
            SCO2 = -R_pt[ix] * M_gas[1]; SCO = R_co2r[ix] * M_gas[2]; SH2 = R_her[ix] * M_gas[3]
        end
        F[gas_resid_idx(ly, cc, 1)] = Jg[1, cc] - Jg[1, cc+1] + SCO2 * dxg[cc]
        F[gas_resid_idx(ly, cc, 2)] = Jg[2, cc] - Jg[2, cc+1] + SCO  * dxg[cc]
        F[gas_resid_idx(ly, cc, 3)] = Jg[3, cc] - Jg[3, cc+1] + SH2  * dxg[cc]
        F[gas_resid_idx(ly, cc, 4)] = rhou[cc] - rhou[cc+1] + (SCO2 + SCO + SH2) * dxg[cc]
    end

    # ── φ_L residual — LITERAL §3.1: five-species ionic current, no Cs⁺ term ──
    for ix in 1:N
        iL_R = F_const * (z_charge[1]*J[1,ix+1] + z_charge[2]*J[2,ix+1] + z_charge[3]*J[3,ix+1] +
                          z_charge[4]*J[4,ix+1] + z_charge[5]*J[5,ix+1])
        iL_L = F_const * (z_charge[1]*J[1,ix]   + z_charge[2]*J[2,ix]   + z_charge[3]*J[3,ix]   +
                          z_charge[4]*J[4,ix]   + z_charge[5]*J[5,ix])
        i_far = n_e * F_const * (R_co2r[ix] + R_her[ix])
        F[phi_L_idx5(ly, ix)] = (iL_R - iL_L) + i_far * dx[ix]
    end
    F[phi_L_idx5(ly, N)] = phi_L[N]                  # §3.1 gauge override at the electrolyte cell

    # ── φ_s residual (identical to production) ──
    i_s = zeros(ElT, Ng+1)
    i_s[1] = -sigma_s_eff_GDL * (phi_s[1] - V_applied_SHE) / gas_mesh.dx_face_left
    for f in 2:Ng
        σf = harmonic_mean(sigma_solid_cell(ly, f-1), sigma_solid_cell(ly, f))
        i_s[f] = -σf * (phi_s[f] - phi_s[f-1]) / gas_mesh.dx_between[f-1]
    end
    for ig in 1:NGDL
        F[phi_s_gdl_idx5(ly, ig)] = i_s[ig+1] - i_s[ig]
    end
    for ix in 1:N
        cc = NGDL + ix
        i_far = n_e * F_const * (R_co2r[ix] + R_her[ix])
        F[phi_s_cl_idx5(ly, ix)] = (i_s[cc+1] - i_s[cc]) - i_far * dx[ix]
    end
    return F
end

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
function dense_jacobian(res!, u::Vector{Float64})
    N = length(u); Jd = zeros(N, N); Fb = zeros(N)
    cfg = ForwardDiff.JacobianConfig(res!, Fb, u, ForwardDiff.Chunk{min(12, N)}())
    ForwardDiff.jacobian!(Jd, res!, Fb, u, cfg)
    return Jd, Fb
end

function svd_report(label, res!, u)
    Jd, _ = dense_jacobian(res!, u)
    s = svdvals(Jd)
    nnull = count(x -> x < 1e-10 * s[1], s)
    @printf("  %-58s σ_max=%.3e  σ_min=%.3e  cond=%.3e  #(σ<1e-10·σ_max)=%d\n",
            label, s[1], s[end], s[1]/s[end], nnull)
    return s
end

function identity_check(res!, u, ly, N)
    F = zeros(length(u)); res!(F, u)
    worst = 0.0; scale = 0.0
    for ix in 1:N-1                                   # cell N carries the gauge override
        lhs = F[phi_L_idx5(ly, ix)]
        rhs = -F_const * sum(z_charge[i] * F[liquid_idx(ix, i)] for i in 1:5)
        worst = max(worst, abs(lhs - rhs)); scale = max(scale, abs(lhs))
    end
    return worst, scale
end

# raw (unclamped) Newton correction at u: max |Δu| over the liquid log block and overall
function raw_newton_step(res!, u, nlog)
    Jd, F = dense_jacobian(res!, u)
    du = try
        lu(Jd) \ (-F)
    catch e
        return (NaN, NaN, "dense LU threw $(typeof(e))")
    end
    return (maximum(abs, view(du, 1:nlog)), maximum(abs, du), all(isfinite, du) ? "finite" : "non-finite")
end

fmt_info(info) = @sprintf("converged=%-5s it=%3d reason=%-18s err_est=%.3e θ=%.3e α=%.3e ‖F‖∞=%.3e n_res=%d",
                          string(info.converged), info.iterations, info.reason,
                          info.err_est, info.theta, info.alpha_final, info.residual, info.n_res)

# ─────────────────────────────────────────────────────────────────────────────
function main()
    load_fitted!()
    cl_mesh  = make_cl_mesh(N_CL, L_CL; q = 1.3)
    gdl_mesh = make_gdl_mesh(N_GDL, L_GDL; q = 1.3)
    gas_mesh = make_gas_mesh(gdl_mesh, cl_mesh)
    ly = GasLayout(N_CL, N_GDL)
    c_eq_vec = equilibrium_vec(1000.0, 1000.0)
    Ndof = ndof_stage5(ly); nlog = 5*N_CL
    offset = V_RHE_to_SHE_offset
    blocks = dof_blocks(5; N_CL_ = ly.N_CL, N_GDL_ = ly.N_GDL)

    println("="^78)
    println("=== EXPERIMENT: Stage 5 with Cs⁺ OMITTED from i_L (literal §3.1) on the §9.5 stack ===")
    @printf("FITTED kinetics: i0=%.2f A/m², λ=%.4f eV, HER_pf=%.4e, α_c=%.5f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    @printf("Mesh N_CL=%d N_GDL=%d, DOFs=%d\n", N_CL, N_GDL, Ndof)
    println("="^78)

    V_ref = Ref(offset)
    res_lit!  = (F_, u_) -> residual_stage5_literal!(F_, u_, V_ref[], cl_mesh, gas_mesh, c_eq_vec, ly)
    res_prod! = (F_, u_) -> residual_stage5!(F_, u_, V_ref[], cl_mesh, gas_mesh, c_eq_vec, ly)

    # ── A. residual at the anchor guess ──
    u_anchor = make_u0_stage5_anchor(ly, c_eq_vec)
    V_ref[] = offset
    F0 = zeros(Ndof); res_lit!(F0, u_anchor)
    F0p = zeros(Ndof); res_prod!(F0p, u_anchor)
    println("\n[A] residual at the constructed V_c = 0 anchor guess (equilibrium liquid, inlet gas, φ = 0)")
    @printf("  literal    ‖F‖∞ = %.4e  (argmax block: %s)\n", maximum(abs, F0), block_of(argmax(abs.(F0)), blocks))
    @printf("  production ‖F‖∞ = %.4e\n", maximum(abs, F0p))
    iH2_0 = compute_i_H2_s5(u_anchor, ly, cl_mesh)/10; iCO_0 = compute_i_CO_s5(u_anchor, ly, cl_mesh)/10
    @printf("  currents at the guess: i_H2 = %.3e mA/cm², i_CO = %.3e mA/cm²\n", iH2_0, iCO_0)

    # ── generic state for B/C: production Stage-5 solution nearest −1.205 V if cached, else perturbed anchor ──
    Vg = -1.2; u_generic = nothing; generic_label = ""
    s5Vs = list_cache(S5_CACHE, "s5")
    if !isempty(s5Vs)
        Vg = s5Vs[argmin(abs.(s5Vs .- (-1.205)))]
        u_generic = load_solution(S5_CACHE, "s5", Vg)
        generic_label = @sprintf("production Stage-5 solution at V_RHE=%.3f", Vg)
    else
        Vg = -1.2
        u_generic = copy(u_anchor); shift_phis!(u_generic, Vg - offset, ly)
        for ix in 1:N_CL; u_generic[phi_L_idx5(ly, ix)] = -0.01 * ix / N_CL; end
        generic_label = @sprintf("perturbed anchor guess at V_RHE=%.3f (no Stage-5 cache found)", Vg)
    end
    u_pert = copy(u_generic)                              # non-equilibrium: perturb lnc and φ_L
    for ix in 1:N_CL, i in 1:5; u_pert[liquid_idx(ix, i)] += 0.05 * sin(1.7*ix + 0.9*i); end
    for ix in 1:N_CL; u_pert[phi_L_idx5(ly, ix)] += 0.003 * cos(0.4*ix); end

    # ── B. algebraic identity of the literal φ_L rows ──
    V_ref[] = Vg
    worst, scale = identity_check(res_lit!, u_pert, ly, N_CL)
    println("\n[B] identity F_φL[ix] = −F·Σ z_i F_species[ix] for the LITERAL residual, cells 1..N−1")
    @printf("  state: %s + lnc/φ_L perturbation\n", generic_label)
    @printf("  max|F_φL + F Σ z F_sp| = %.3e   max|F_φL| = %.3e   relative = %.3e\n", worst, scale, worst/scale)
    worst_p, scale_p = identity_check(res_prod!, u_pert, ly, N_CL)
    @printf("  (production residual, same state: relative mismatch = %.3e — the Cs⁺ term breaks the identity)\n", worst_p/scale_p)

    # ── C. SVD ──
    println("\n[C] dense AD Jacobian SVD (σ < 1e-10·σ_max counted as near-null)")
    V_ref[] = offset
    svd_report("literal    @ anchor guess (V_c = 0)", res_lit!, u_anchor)
    svd_report("production @ anchor guess (V_c = 0)", res_prod!, u_anchor)
    V_ref[] = Vg
    svd_report("literal    @ " * generic_label, res_lit!, u_generic)
    svd_report("production @ " * generic_label, res_prod!, u_generic)
    svd_report("literal    @ perturbed generic state", res_lit!, u_pert)
    fitVs = list_cache(FIT_CACHE, "fit")
    if !isempty(fitVs)
        Vf = fitVs[argmin(abs.(fitVs .- (-1.205)))]
        u_fit = load_solution(FIT_CACHE, "fit", Vf)
        res4! = (F_, u_) -> residual_stage4!(F_, u_, Vf, cl_mesh, gas_mesh, c_eq_vec, ly)
        svd_report(@sprintf("Stage 4 (fit cache) @ V_RHE=%.3f, for comparison", Vf), res4!, u_fit)
    end

    # ── shared §9.5 workspace for the literal residual ──
    perturb = zeros(Ndof)
    for ix in 1:N_CL; perturb[phi_L_idx5(ly, ix)] = 0.005; perturb[phi_s_cl_idx5(ly, ix)] = 0.01; end
    for ig in 1:N_GDL; perturb[phi_s_gdl_idx5(ly, ig)] = 0.01; end
    for cc in 1:(N_GDL+N_CL); perturb[gas_resid_idx(ly, cc, 4)] = -0.1 * cc; end
    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    V_ref[] = offset; r1, c1 = probe_sparsity(res_lit!, u_anchor; perturb = perturb)
    V_ref[] = Vs[1];  r2, c2 = probe_sparsity(res_lit!, u_anchor; perturb = perturb)
    V_ref[] = offset
    Jtemplate = pattern_matrix(vcat(r1, r2), vcat(c1, c2), Ndof)
    assemble! = make_ad_assembler(res_lit!, Ndof; chunk = 12)
    plu = AffineLU()
    kw = (blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jtemplate, n_log_dofs = nlog)

    # ── D. anchor solve ──
    println("\n[D] anchor solve at V_c = 0 on the literal residual (newton_affine!, alpha_init=1e-2, max_iter=200)")
    V_ref[] = offset
    mx_liq, mx_all, note = raw_newton_step(res_lit!, u_anchor, nlog)
    @printf("  raw first Newton correction (dense LU, before the §9.5.3 clamp): max|Δu_liquid| = %.3e log-units, max|Δu| = %.3e  [%s]\n",
            mx_liq, mx_all, note)
    u = copy(u_anchor)
    t = @elapsed info_anchor = newton_affine!(u, res_lit!; kw..., alpha_init = 1e-2, max_iter = 200, verbose = true)
    @printf("  anchor: %s  (%.1f s)\n", fmt_info(info_anchor), t)
    Fchk = zeros(Ndof); res_lit!(Fchk, u)
    @printf("  §9.5.6 assertion ‖F‖∞ ≤ 1e-6 at the anchor: %s (‖F‖∞ = %.3e)\n",
            info_anchor.converged && maximum(abs, Fchk) <= 1e-6 ? "PASS" : "FAIL", maximum(abs, Fchk))

    # ── E. chain ──
    if get(ENV, "CS_TEST_SKIP_CHAIN", "0") != "1"
        n_pre = floor(Int, (offset - Vs[1]) / 0.025 - 1e-9)
        chain = vcat([offset - 0.025*s for s in 0:n_pre], Vs)
        maxpts = parse(Int, get(ENV, "CS_TEST_MAX_POINTS", string(length(chain))))
        chain = chain[1:min(maxpts, length(chain))]
        out_set = Set(round.(Vs, digits=6))
        println("\n[E] continuation chain on the literal residual: $(length(chain)) points, tangent predictor when the previous point converged, φ_s-shift otherwise; carry-forward on failure; no sub-step recovery")
        println("     k   V_RHE     V_c    out?  conv  it  reason              err_est    θ          α          ‖F‖∞       assert")
        dudV = zeros(Ndof); Fp = zeros(Ndof); Fm = zeros(Ndof)
        res_at_V!(F_, uu, V) = (Vold = V_ref[]; V_ref[] = V; res_lit!(F_, uu); V_ref[] = Vold; F_)
        u_prev = copy(u)
        tangent_ok = info_anchor.converged && tangent_predictor!(dudV, plu, res_at_V!, u, chain[1]; Fp = Fp, Fm = Fm)
        reasons = Dict{String,Int}(); n_ok_out = 0; n_out = 0; n_ok_all = 0; Fexit = Float64[]
        t_chain = time()
        for k in 2:length(chain)
            u_prev .= u
            dV = chain[k] - chain[k-1]
            tangent_ok ? (u .+= dV .* dudV) : shift_phis!(u, dV, ly)
            V_ref[] = chain[k]
            info = newton_affine!(u, res_lit!; kw..., alpha_init = 1.0, max_iter = 40)
            res_lit!(Fchk, u); r = maximum(abs, Fchk)
            ok = info.converged && r <= 1e-6
            isout = round(chain[k], digits=6) in out_set
            @printf("  %4d  %+.3f  %+.3f   %-4s  %-5s %3d  %-18s  %.2e  %.2e  %.2e  %.2e  %s\n",
                    k, chain[k], chain[k]-offset, isout ? "yes" : "-", string(info.converged), info.iterations,
                    info.reason, info.err_est, info.theta, info.alpha_final, r, ok ? "pass" : "FAIL")
            reasons[info.reason] = get(reasons, info.reason, 0) + 1
            push!(Fexit, r); n_ok_all += ok
            if isout; n_out += 1; n_ok_out += ok; end
            if ok
                tangent_ok = tangent_predictor!(dudV, plu, res_at_V!, u, chain[k]; Fp = Fp, Fm = Fm)
            else
                u .= u_prev; tangent_ok = false
            end
        end
        @printf("\n  chain summary: %d/%d output voltages passed, %d/%d chain steps passed, %.1f s\n",
                n_ok_out, n_out, n_ok_all, length(chain)-1, time()-t_chain)
        println("  exit reasons: ", join(["$k×$v" for (v, k) in reasons], ", "))
        if !isempty(Fexit)
            @printf("  exit ‖F‖∞ over the chain: min %.3e  median %.3e  max %.3e\n",
                    minimum(Fexit), sort(Fexit)[(length(Fexit)+1)÷2], maximum(Fexit))
        end
    end

    # ── F. fit-cache warm-starts ──
    println("\n[F] newton_affine! from fit-cache warm-starts (Stage-4 fit state + φ_s = V_app, φ_L = 0), literal residual, alpha_init=1e-2, max_iter=200")
    if isempty(fitVs)
        println("  (no output_fit/cache — skipped)")
    else
        for Vexp in E_exp_RHE
            Vf = fitVs[argmin(abs.(fitVs .- Vexp))]
            u_fit = load_solution(FIT_CACHE, "fit", Vf)
            u0 = zeros(Ndof); u0[1:ndof_stage3(ly)] .= u_fit
            shift_phis!(u0, Vf - offset, ly)               # φ_s = V_applied_SHE everywhere, φ_L = 0
            V_ref[] = Vf
            F0 = zeros(Ndof); res_lit!(F0, u0)
            uu = copy(u0)
            t = @elapsed info = newton_affine!(uu, res_lit!; kw..., alpha_init = 1e-2, max_iter = 200)
            iCO = compute_i_CO_s5(uu, ly, cl_mesh)/10; iH2 = compute_i_H2_s5(uu, ly, cl_mesh)/10
            iCOf = compute_i_CO(u_fit, Vf - offset, cl_mesh)/10; iH2f = compute_i_H2(u_fit, Vf - offset, cl_mesh)/10
            dphiL = (uu[phi_L_idx5(ly, 1)] - uu[phi_L_idx5(ly, N_CL)]) * 1000
            @printf("  V_RHE=%.3f: ‖F‖∞(start)=%.3e  %s  (%.1f s)\n", Vf, maximum(abs, F0), fmt_info(info), t)
            @printf("             partial solution: i_CO=%.1f (fit %.1f) i_H2=%.1f (fit %.1f) mA/cm², Δφ_L=%+.2f mV\n",
                    iCO, iCOf, iH2, iH2f, dphiL)
        end
    end
    println("\n=== experiment complete ===")
end

block_of(j, blocks) = (names = ("liquid", "gas GDL", "gas CL", "φ_s GDL", "φ_s CL", "φ_L");
                       k = findfirst(b -> j in b, blocks); k === nothing ? "?" : names[k])

main()
