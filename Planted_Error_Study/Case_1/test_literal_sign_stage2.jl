# test_literal_sign_stage2.jl — Stage-2 Sherwood-sign experiment (BUILD_DIALOGUE.md, Stage 2 gate).
#
# Re-measures, with the §9.5 stack, the convergence signature of the LITERAL planted form of the
# electrolyte-face flux written in §7.2 / §11 Stage 2,
#     J[i,N+1] = k_MT[i] * (c_eq[i] − c[i,N])          (literal)
# against the corrected np_flux-consistent form used in production (assembly.jl),
#     J[i,N+1] = k_MT[i] * (c[i,N] − c_eq[i])          (corrected)
#
# The production residual `residual_stage2!` is NOT modified. This script defines a test-local copy
# whose right-face sign is selected by the flag LITERAL_SIGN, verifies that the corrected copy
# reproduces the production residual bit-for-bit, and then runs BOTH signs through
#   (a) per-voltage Newton (`newton_affine!`) with carry-forward warm starts and a cautious retry
#       from the equilibrium guess — the per-voltage signature at every grid point, and
#   (b) the §9.5.5 continuation driver (`continuation_sweep!`), i.e. exactly the production path.
# Prints, per voltage: converged / reason / iterations / ‖F‖∞ / err_est / c_CO2(L_CL) / c_OH(L_CL)/bulk,
# then a summary block for the dialogue's comparison table. No caches or plots are written.
#
# Run from the Case_1 folder:   julia test_literal_sign_stage2.jl     (MESH_N_CL env override honoured)

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "solver.jl"))
include(joinpath(HERE, "cache.jl"))
include(joinpath(HERE, "assembly.jl"))

const N_CL = parse(Int, get(ENV, "MESH_N_CL", "80"))
const LITERAL_SIGN = Ref(false)      # test-local flag: true → planted literal form of the right-face flux
const F_INF_TOL = 1e-6               # §9.5.6 post-solve assertion level

"""
Test-local copy of `residual_stage2!` (assembly.jl). Identical line for line except the
electrolyte-face flux, whose sign is chosen by LITERAL_SIGN[].
"""
function residual_stage2_signed!(F::AbstractVector{<:Real}, u::AbstractVector{<:Real},
                                 V_cathode::Real, cl_mesh, c_eq::AbstractVector)
    ElT = eltype(u)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    J = zeros(ElT, 5, N + 1)
    J[5, 1] = diffusion_flux(c_CO2_henry, c[5, 1], D_eff[5], cl_mesh.dx_face_left)  # CO₂ Henry BC
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl_mesh.dx_cc[f-1])
        end
    end
    # Right face: Sherwood-Reynolds convective Neumann — sign under test
    if LITERAL_SIGN[]
        for i in 1:5
            J[i, N+1] = k_MT[i] * (c_eq[i] - c[i, N])          # literal §7.2 / §11 Stage-2 form
        end
    else
        for i in 1:5
            J[i, N+1] = k_MT[i] * (c[i, N] - c_eq[i])          # corrected (production) form
        end
    end

    for ix in 1:N
        c_ix = @view c[:, ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, zero(ElT), c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end
            if i == 5; src -= r_co2r;             end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end
    return F
end

fmt_reason(info) = info.converged ? (info.residual <= F_INF_TOL ? "conv" : "conv(assert-fail)") : info.reason

function main()
    println("="^78)
    println("=== Stage-2 Sherwood-sign experiment: literal §7.2/§Stage-2 form vs corrected form ===")
    @printf("Mesh N_CL=%d (DOFs=%d); kinetics i0=%.0f λ=%.2f HER_pf=%.3e α_c=%.4f\n",
            N_CL, 5*N_CL, i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    println("="^78)

    cl_mesh = make_cl_mesh(N_CL)
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    ND = 5 * N_CL
    Vs = sort(collect(V_RHE_grid), by = v -> -v)                  # −0.300 → −2.500
    u_eq = repeat(log.(c_eq), N_CL)

    # ── 0. the corrected copy must reproduce the production residual bit-for-bit ──
    LITERAL_SIGN[] = false
    rng_u = u_eq .+ 0.3 .* sin.(1.0:ND)                            # deterministic off-equilibrium state
    Fa = zeros(ND); Fb = zeros(ND)
    residual_stage2!(Fa, rng_u, -1.5, cl_mesh, c_eq)
    residual_stage2_signed!(Fb, rng_u, -1.5, cl_mesh, c_eq)
    @assert Fa == Fb "test copy (corrected sign) does not reproduce production residual_stage2!"
    LITERAL_SIGN[] = true
    residual_stage2_signed!(Fb, rng_u, -1.5, cl_mesh, c_eq)
    @assert Fa != Fb "literal flag has no effect"
    println("check: corrected-sign test copy ≡ production residual_stage2! (bitwise) ✓; literal flag changes the residual ✓\n")

    # ── §9.5 workspace shared by both signs (the flag is read inside the closure) ──
    V_ref = Ref(Vs[1])
    res!(F, uu) = residual_stage2_signed!(F, uu, V_ref[], cl_mesh, c_eq)
    rows, cols = block_tridiag_pattern(N_CL, 5)
    Jtemplate = pattern_matrix(rows, cols, ND)
    assemble! = make_ad_assembler(res!, ND; chunk=12)
    blocks = dof_blocks(2, N_CL)
    kw = (blocks=blocks, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=ND)

    summary = Dict{String,Any}()

    # ── (a) per-voltage Newton, carry-forward warm start, retry from equilibrium with alpha_init=1e-2 ──
    for (label, literal) in (("literal  k_MT·(c_eq−c[N])", true), ("corrected k_MT·(c[N]−c_eq)", false))
        LITERAL_SIGN[] = literal
        plu = AffineLU()
        println("─"^78)
        println("(a) per-voltage newton_affine!  —  $label")
        println("    V_RHE   conv/reason          it   ||F||inf    err_est    c_CO2(L)     c_OH(L)/bulk   pH(L)")
        u = copy(u_eq)
        nconv = 0; Fs_fail = Float64[]; reasons = Dict{String,Int}(); cco2 = Float64[]; coh = Float64[]; pHL = Float64[]; Fs_ok = Float64[]
        Fchk = zeros(ND)
        for V in Vs
            V_ref[] = V
            u_try = copy(u)
            info = newton_affine!(u_try, res!; kw..., plu=plu, alpha_init=1.0, max_iter=40)
            ok = info.converged && info.residual <= F_INF_TOL
            if !ok
                u_try = copy(u_eq)
                info = newton_affine!(u_try, res!; kw..., plu=plu, alpha_init=1e-2, max_iter=200)
                ok = info.converged && info.residual <= F_INF_TOL
            end
            res!(Fchk, u_try); Finf = maximum(abs, Fchk)
            c = unpack_log_conc(u_try, N_CL)
            r = fmt_reason(info)
            @printf("  %7.3f   %-18s %4d   %.2e   %.2e   %.3e   %10.2f   %.2f\n",
                    V, r, info.iterations, Finf, info.err_est, c[5,end], c[2,end]/c_eq[2], -log10(c[1,end]/1000))
            ok ? (nconv += 1; push!(Fs_ok, Finf)) : (push!(Fs_fail, Finf); reasons[r] = get(reasons, r, 0) + 1)
            push!(cco2, c[5,end]); push!(coh, c[2,end]/c_eq[2]); push!(pHL, -log10(c[1,end]/1000))
            u = u_try                                          # carry the final iterate forward
        end
        summary[label*" (a)"] = (nconv=nconv, Fs_fail=Fs_fail, Fs_ok=Fs_ok, reasons=reasons, cco2=cco2, coh=coh, pHL=pHL)
    end

    # ── (b) the §9.5.5 continuation driver — the production path ──
    for (label, literal) in (("literal  k_MT·(c_eq−c[N])", true), ("corrected k_MT·(c[N]−c_eq)", false))
        LITERAL_SIGN[] = literal
        plu = AffineLU()
        println("─"^78)
        println("(b) continuation_sweep! (§9.5.5, anchor −0.300 V from equilibrium)  —  $label")
        u = copy(u_eq)
        cco2 = Float64[]; coh = Float64[]; pHL = Float64[]
        outcome = ""
        log = NamedTuple[]
        try
            log = continuation_sweep!(u, Vs, res!, V_ref; kw..., plu=plu,
                on_converged = (V, uu, info) -> begin
                    c = unpack_log_conc(uu, N_CL)
                    push!(cco2, c[5,end]); push!(coh, c[2,end]/c_eq[2]); push!(pHL, -log10(c[1,end]/1000))
                end)
            nconv = count(l -> l.converged, log)
            maxF = maximum(l.residual for l in log if l.converged; init=0.0)
            outcome = @sprintf("swept %d chain points, %d/%d converged, max ||F||inf (converged) = %.2e, %d Newton its (%.2f/step), %d sub-step recoveries",
                               length(log), nconv, length(Vs), maxF, sum(l.iterations for l in log), sum(l.iterations for l in log)/length(log), sum(l.substeps for l in log))
        catch e
            outcome = "driver aborted: " * sprint(showerror, e)
        end
        println("  outcome: ", outcome)
        summary[label*" (b)"] = (outcome=outcome, cco2=cco2, coh=coh, pHL=pHL)
    end

    # ── summary for the BUILD_DIALOGUE.md comparison table ──
    println("═"^78)
    println("SUMMARY (fill the «C1_lit_*» / «C1_s2_*» tokens from these lines)")
    for label in ("literal  k_MT·(c_eq−c[N])", "corrected k_MT·(c[N]−c_eq)")
        a = summary[label*" (a)"]; b = summary[label*" (b)"]
        println("── $label")
        @printf("  (a) per-voltage: %d/%d converged (converged = err_est ≤ 1e-8 AND ||F||inf ≤ 1e-6)\n", a.nconv, length(Vs))
        if !isempty(a.Fs_fail)
            @printf("      failures: ||F||inf at exit %.2e – %.2e; reasons %s\n", minimum(a.Fs_fail), maximum(a.Fs_fail), a.reasons)
        end
        if !isempty(a.Fs_ok)
            @printf("      converged: ||F||inf %.1e – %.1e\n", minimum(a.Fs_ok), maximum(a.Fs_ok))
        end
        @printf("      c_CO2(L_CL): %.3e → %.3e (mild → cathodic; min %.3e, max %.3e)\n", a.cco2[1], a.cco2[end], minimum(a.cco2), maximum(a.cco2))
        @printf("      c_OH(L_CL)/bulk: %.1f× → %.0f×;  pH(L_CL): %.2f → %.2f\n", a.coh[1], a.coh[end], a.pHL[1], a.pHL[end])
        println("  (b) §9.5.5 driver: ", b.outcome)
        if !isempty(b.cco2)
            @printf("      converged points: c_CO2(L_CL) %.3e → %.3e;  c_OH(L_CL)/bulk %.1f× → %.0f×;  pH(L_CL) %.2f → %.2f\n",
                    b.cco2[1], b.cco2[end], b.coh[1], b.coh[end], b.pHL[1], b.pHL[end])
        end
    end
    println("═"^78)
    LITERAL_SIGN[] = false
    return summary
end

main()
