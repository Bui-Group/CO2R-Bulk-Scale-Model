# driver.jl — the §9.5.5 unified continuation driver, shared by ALL stages.
# Anchor → tangent-predictor sweep → §9.5.6 post-solve assertion → cache.
# Written ONCE; five per-stage copies diverge.

const ROOT = @__DIR__

struct StageRun{RF,CF,VS}
    stage::Int
    ly::Layout
    cl::Mesh1D
    gdl::Mesh1D
    geom::GasGeom
    c_eq::Vector{Float64}
    Vref::Base.RefValue{Float64}
    ws::NewtonWS{RF,CF}
    valS::VS
    cache_dir::String
    out_dir::String
    prefix::String
end

function make_stage_run(::Val{S}, N_CL::Int, N_GDL::Int;
                        outname::String="output_stage$(S)",
                        prefix::String="s$(S)") where {S}
    cl   = make_cl_mesh(N_CL)
    gdl  = make_gdl_mesh(N_GDL)
    geom = make_gas_geom(cl, gdl)
    ly   = Layout(N_CL, N_GDL, S)
    c_eq = equilibrium_vector()
    Vref = Ref(-0.300)
    valS = Val(S)
    res! = (Fv, uv) -> full_residual!(Fv, uv, Vref[], cl, gdl, c_eq, ly, geom, valS)
    u0   = build_initial_guess(ly, cl, gdl, c_eq, 0.0)
    ws   = NewtonWS(ly.N_TOT, dof_blocks(ly), res!, u0)
    out_dir   = joinpath(ROOT, outname)
    cache_dir = joinpath(out_dir, "cache")
    mkpath(cache_dir)
    return StageRun(S, ly, cl, gdl, geom, c_eq, Vref, ws, valS,
                    cache_dir, out_dir, prefix)
end

"""
    build_initial_guess(ly, cl, gdl, c_eq, V_applied_SHE)

Anchor guess constructed from physics (§9.5.5): every liquid cell at the exact
bulk equilibrium, every gas cell at the inlet composition and `p_total`, and
(Stage 5) `φ_s = V_applied_SHE` on GDL+CL with `φ_L = 0` on the CL.
"""
function build_initial_guess(ly::Layout, cl::Mesh1D, gdl::Mesh1D,
                             c_eq::Vector{Float64}, V_applied_SHE::Float64)
    u = zeros(ly.N_TOT)
    lnc = log.(c_eq)
    for ix in 1:ly.N_CL, i in 1:5
        u[liquid_idx(ly, ix, i)] = lnc[i]
    end
    if ly.stage >= 3
        for k in 1:(ly.N_GDL + ly.N_CL)
            u[gas_idx(ly, k, 1)] = omega_inlet[1]
            u[gas_idx(ly, k, 2)] = omega_inlet[2]
            u[gas_idx(ly, k, 3)] = omega_inlet[3]
            u[gas_idx(ly, k, 4)] = p_total
        end
    end
    if ly.stage >= 5
        for ig in 1:ly.N_GDL; u[phi_s_gdl_idx(ly, ig)] = V_applied_SHE; end
        for ix in 1:ly.N_CL;  u[phi_s_cl_idx(ly, ix)]  = V_applied_SHE; end
        for ix in 1:ly.N_CL;  u[phi_L_idx(ly, ix)]     = 0.0;           end
    end
    return u
end

# ── §10.3 binary cache ────────────────────────────────────────────────────
function save_stage(run::StageRun, u::Vector{Float64}, V::Float64; quiet::Bool=true)
    mkpath(run.cache_dir)
    fn = joinpath(run.cache_dir, @sprintf("%s_V%.6f.bin", run.prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    quiet || @printf("  [CACHED] %s (%d DOFs)\n", fn, length(u))
    return fn
end

function load_stage(run::StageRun, V::Float64)
    fn = joinpath(run.cache_dir, @sprintf("%s_V%.6f.bin", run.prefix, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

function list_cache(cache_dir::String, prefix::String)
    isdir(cache_dir) || return Float64[]
    files = filter(f -> startswith(f, prefix * "_V") && endswith(f, ".bin"),
                   readdir(cache_dir))
    return sort([parse(Float64, f[length(prefix)+3:end-4]) for f in files])
end

# ── residual helper at an arbitrary V (used by the predictor and the check) ─
function res_at_V!(run::StageRun, F::Vector{Float64}, u::Vector{Float64}, V::Float64)
    old = run.Vref[]; run.Vref[] = V
    run.ws.res!(F, u)
    run.Vref[] = old
    return F
end

const F_INF_TOL = 1e-6

"""
    post_solve_check!(run, u, V, Fchk; n_log_dofs) -> ‖F‖∞

§9.5.6 post-solve residual assertion.  Outside the termination test on purpose.
"""
function post_solve_check!(run::StageRun, u::Vector{Float64}, V::Float64,
                           Fchk::Vector{Float64}; n_log_dofs::Int=0)
    res_at_V!(run, Fchk, u, V)
    r = maximum(abs, Fchk)
    if r > F_INF_TOL
        run.Vref[] = V
        newton_affine!(u, run.ws; rtol=1e-10, max_iter=3, alpha_init=1.0,
                       n_log_dofs=n_log_dofs)
        res_at_V!(run, Fchk, u, V)
        r = maximum(abs, Fchk)
        r > F_INF_TOL && error("residual assertion failed at V=$V (‖F‖∞=$r) — halve dV and re-approach")
        @printf("    [assertion fired and recovered at V=%.4f → ‖F‖∞=%.2e]\n", V, r)
    end
    return r
end

"φ_s-shift fallback predictor (§9.5.4): the boundary value moves by exactly dV."
function phis_shift!(u::Vector{Float64}, ly::Layout, dV::Float64)
    ly.stage >= 5 || return u
    for ig in 1:ly.N_GDL; u[phi_s_gdl_idx(ly, ig)] += dV; end
    for ix in 1:ly.N_CL;  u[phi_s_cl_idx(ly, ix)]  += dV; end
    return u
end

"""
    run_sweep!(run, u_anchor, anchor_V, chain, emit; ...) -> (Vs, us, stats)

Anchor once, then walk `chain` with the tangent predictor.  Voltages present in
`emit` are asserted, cached, and returned.  Sub-step recovery halves dV and
re-approaches at `alpha_init = 1e-2`.
"""
function run_sweep!(run::StageRun, u_anchor::Vector{Float64}, anchor_V::Float64,
                    chain::Vector{Float64}, emit::Vector{Float64};
                    rtol::Float64=1e-8, anchor_alpha::Float64=1e-2,
                    anchor_max_iter::Int=200, verbose::Bool=true)
    ly = run.ly
    n_log = 5 * ly.N_CL
    N = ly.N_TOT
    Fchk = zeros(N)
    emitset = Set(round.(emit, digits=9))
    isemit(V) = round(V, digits=9) in emitset

    u = copy(u_anchor)
    run.Vref[] = anchor_V
    t0 = time()
    info = newton_affine!(u, run.ws; rtol=rtol, max_iter=anchor_max_iter,
                          alpha_init=anchor_alpha, n_log_dofs=n_log)
    if !info.converged
        error("ANCHOR FAILED at V=$anchor_V: $(info.reason), ‖F‖∞=$(info.residual)")
    end
    verbose && @printf("anchor  V=%8.4f  iters=%3d  ‖F‖∞=%.3e  err=%.2e\n",
                       anchor_V, info.iterations, info.residual, info.err_est)

    Vs = Float64[]; us = Vector{Vector{Float64}}(); iters = Int[]; resn = Float64[]
    if isemit(anchor_V)
        r = post_solve_check!(run, u, anchor_V, Fchk; n_log_dofs=n_log)
        save_stage(run, u, anchor_V)
        push!(Vs, anchor_V); push!(us, copy(u)); push!(iters, info.iterations); push!(resn, r)
    end

    dudV = zeros(N)
    have_tan = tangent_predictor!(dudV, run.ws, (F,uu,V)->res_at_V!(run,F,uu,V), u, anchor_V)
    Vprev = anchor_V
    u_prev = copy(u)

    total_iters = info.iterations
    nsub = 0

    function take_step!(Vtarget::Float64, dV::Float64, depth::Int, alpha0::Float64)
        # predictor
        if have_tan
            @inbounds for j in 1:N; u[j] += dV * dudV[j]; end
        else
            phis_shift!(u, ly, dV)
        end
        run.Vref[] = Vtarget
        inf2 = newton_affine!(u, run.ws; rtol=rtol, max_iter=40,
                              alpha_init=alpha0, n_log_dofs=n_log)
        if inf2.converged
            return inf2
        end
        depth >= 4 && error("step failed at V=$Vtarget after $depth bisections: $(inf2.reason)")
        # sub-step recovery: roll back, halve dV
        nsub += 1
        copyto!(u, u_prev)
        Vh = Vprev + dV/2
        i1 = take_step!(Vh, dV/2, depth+1, 1e-2)
        copyto!(u_prev, u); Vprev = Vh
        have_tan = tangent_predictor!(dudV, run.ws, (F,uu,V)->res_at_V!(run,F,uu,V), u, Vh)
        i2 = take_step!(Vtarget, dV/2, depth+1, 1e-2)
        return i2
    end

    for (istep, V) in enumerate(chain)
        dV = V - Vprev
        inf2 = take_step!(V, dV, 0, 1.0)
        total_iters += inf2.iterations
        if isemit(V)
            r = post_solve_check!(run, u, V, Fchk; n_log_dofs=n_log)
            save_stage(run, u, V)
            push!(Vs, V); push!(us, copy(u)); push!(iters, inf2.iterations); push!(resn, r)
            verbose && @printf("step %3d V=%8.4f iters=%2d ‖F‖∞=%.2e err=%.2e θ=%.2e\n",
                               istep, V, inf2.iterations, r, inf2.err_est, inf2.theta)
        end
        copyto!(u_prev, u); Vprev = V
        have_tan = tangent_predictor!(dudV, run.ws, (F,uu,V2)->res_at_V!(run,F,uu,V2), u, V)
    end
    wall = time() - t0
    verbose && @printf("\nsweep done: %d emitted, %d chain pts, %d sub-steps, %.1f s, %.2f iters/step\n",
                       length(Vs), length(chain), nsub, wall, total_iters/(length(chain)+1))
    return (Vs=Vs, us=us, iters=iters, resn=resn, wall=wall, nsub=nsub)
end

# ══════════════════════════════════════════════════════════════════════════
#  Diagnostics / exports for plotting
# ══════════════════════════════════════════════════════════════════════════
function write_mat(path::String, M::AbstractMatrix{Float64})
    open(path, "w") do f
        write(f, Int64(size(M,1))); write(f, Int64(size(M,2)))
        write(f, Array{Float64}(M))
    end
end
write_vec(path::String, v::AbstractVector{Float64}) = write_mat(path, reshape(collect(v), :, 1))

"Solid-phase face currents at a converged Stage-5 state (Float64 path)."
function solid_currents(run::StageRun, u::Vector{Float64}, V_cathode::Float64)
    is = zeros(run.ly.N_GDL + run.ly.N_CL + 1)
    solid_face_currents!(is, u, run.ly, run.cl, run.gdl,
                         V_cathode - V_RHE_to_SHE_offset)
    return is
end

"""
    export_stage(run, Vs, us) -> summary NamedTuple

Writes summary.csv plus the profile binaries required by §12.7 and returns the
summary columns for in-Julia checks.
"""
function export_stage(run::StageRun, Vs::Vector{Float64}, us::Vector{Vector{Float64}})
    ly, cl, gdl = run.ly, run.cl, run.gdl
    nV = length(Vs); N = ly.N_CL; NK = ly.N_GDL + ly.N_CL
    od = run.out_dir; mkpath(od)

    iCO = zeros(nV); iH2 = zeros(nV)
    pH_m   = zeros(nV, N); cCO2_m = zeros(nV, N); cOH_m = zeros(nV, N)
    cHCO3_m= zeros(nV, N); cCO3_m = zeros(nV, N); cCs_m = zeros(nV, N)
    cH_m   = zeros(nV, N); Rco_m  = zeros(nV, N); Rher_m= zeros(nV, N)
    pCO2_m = zeros(nV, NK); pCO_m = zeros(nV, NK); pH2_m = zeros(nV, NK); ptot_m = zeros(nV, NK)
    phis_m = zeros(nV, NK); phiL_m = zeros(nV, N)
    dphiL = zeros(nV); dphis = zeros(nV); dp_CL = zeros(nV)
    chg_solid = zeros(nV); chg_ionic = zeros(nV)

    for (k, V) in enumerate(Vs)
        u = us[k]
        c = unpack_log_conc(u, ly)
        Rco, Rher = compute_rates(u, ly, cl, V)
        iCO[k] = n_e*F_const*sum(Rco .* cl.dx)
        iH2[k] = n_e*F_const*sum(Rher .* cl.dx)
        for ix in 1:N
            pH_m[k,ix]   = -log10(max(c[1,ix]/1000.0, 1e-30))
            cH_m[k,ix]   = c[1,ix]; cOH_m[k,ix] = c[2,ix]
            cHCO3_m[k,ix]= c[3,ix]; cCO3_m[k,ix]= c[4,ix]; cCO2_m[k,ix] = c[5,ix]
            cCs_m[k,ix]  = cCs_from_electroneutrality(c[1,ix],c[2,ix],c[3,ix],c[4,ix])
            Rco_m[k,ix]  = Rco[ix]; Rher_m[k,ix] = Rher[ix]
        end
        if ly.stage >= 3
            for kk in 1:NK
                w = gas_omega(u, ly, kk)
                Mn = Mn_from_omega(w[1],w[2],w[3],w[4])
                p  = gas_p(u, ly, kk)
                pCO2_m[k,kk] = omega_to_x(w[1],Mn,1)*p/101325.0
                pCO_m[k,kk]  = omega_to_x(w[2],Mn,2)*p/101325.0
                pH2_m[k,kk]  = omega_to_x(w[3],Mn,3)*p/101325.0
                ptot_m[k,kk] = p
            end
            dp_CL[k] = ptot_m[k, ly.N_GDL+1] - ptot_m[k, NK]
        end
        if ly.stage >= 5
            for ig in 1:ly.N_GDL; phis_m[k,ig] = u[phi_s_gdl_idx(ly,ig)]; end
            for ix in 1:N; phis_m[k,ly.N_GDL+ix] = u[phi_s_cl_idx(ly,ix)]; end
            for ix in 1:N; phiL_m[k,ix] = u[phi_L_idx(ly,ix)]; end
            dphiL[k] = phiL_m[k,1] - phiL_m[k,N]
            dphis[k] = phis_m[k,NK] - (V - V_RHE_to_SHE_offset)
            is = solid_currents(run, u, V)
            i_far = iCO[k] + iH2[k]
            chg_solid[k] = abs(abs(is[1]) - i_far) / max(i_far, 1e-30)
            # ionic current at the right boundary (includes migration)
            phiL = phiL_m[k,:]
            iR = 0.0
            for i in 1:5
                L_eff = D_eff[i]/k_MT[i]
                Ji = np_flux(c[i,N], run.c_eq[i], phiL[N], 0.0, D_eff[i], z_sp[i], L_eff)
                iR += z_sp[i]*Ji
            end
            iR *= F_const
            chg_ionic[k] = abs(abs(i_far) - abs(iR)) / max(abs(i_far), 1e-30)
        end
    end
    itot = iCO .+ iH2
    FE_CO = 100 .* iCO ./ max.(itot, 1e-30)

    open(joinpath(od, "summary.csv"), "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO_pct,FE_H2_pct,dphiL_mV,dphis_mV,dp_CL_Pa,chg_solid,chg_ionic")
        for k in 1:nV
            @printf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f,%.4f,%.3e,%.3e\n",
                    Vs[k], Vs[k]-V_RHE_to_SHE_offset, iCO[k]/10, iH2[k]/10, itot[k]/10,
                    FE_CO[k], 100-FE_CO[k], dphiL[k]*1000, dphis[k]*1000, dp_CL[k],
                    chg_solid[k], chg_ionic[k])
        end
    end

    write_vec(joinpath(od,"x_cl_um.bin"), cl.x_centers .* 1e6)
    for (nm, M) in (("pH",pH_m),("cCO2",cCO2_m),("cOH",cOH_m),("cHCO3",cHCO3_m),
                    ("cCO3",cCO3_m),("cCs",cCs_m),("cH",cH_m),("Rco",Rco_m),("Rher",Rher_m))
        write_mat(joinpath(od, "prof_$(nm).bin"), M)
    end
    write_vec(joinpath(od,"V_RHE.bin"), Vs)
    if ly.stage >= 3
        write_vec(joinpath(od,"x_gas_um.bin"), vcat(gdl.x_centers, cl.x_centers) .* 1e6)
        for (nm, M) in (("pCO2",pCO2_m),("pCO",pCO_m),("pH2",pH2_m),("ptot",ptot_m))
            write_mat(joinpath(od, "prof_$(nm).bin"), M)
        end
    end
    if ly.stage >= 5
        write_mat(joinpath(od,"prof_phis.bin"), phis_m)
        write_mat(joinpath(od,"prof_phiL.bin"), phiL_m)
    end
    return (V=Vs, iCO=iCO, iH2=iH2, itot=itot, FE_CO=FE_CO, dphiL=dphiL,
            dphis=dphis, dp_CL=dp_CL, chg_solid=chg_solid, chg_ionic=chg_ionic)
end

"§10.7 stale-cache banner"
function print_kinetics_banner(stage, cache_dir::String)
    @printf("=== Stage %s ===\n", string(stage))
    @printf("i0_CO2R = %.4f A/m², λ_MHC = %.4f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf  = %.6e, α_c = %.6f, k_GL = %.3e\n",
            HER_prefactor(), alpha_c_HER(), k_GL())
    @printf("Cache dir: %s\n", cache_dir)
    if isdir(cache_dir) && !isempty(readdir(cache_dir))
        @printf("WARNING: Existing cache found. Verify kinetics match before reusing!\n")
    end
    flush(stdout)
end
