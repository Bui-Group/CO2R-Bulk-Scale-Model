# run_stage3.jl — Stage 3: Stefan-Maxwell gas transport + R_PT phase transfer (§11 Stage 3)
include("common.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage3", "cache")
const PREFIX    = "s3_"

m = build_model(3)
print_kinetics_header(3, CACHE_DIR)
@printf("mesh: N_CL = %d, N_GDL = %d, DOFs = %d\n", m.cl.N, m.gdl.N, m.ly.N_TOT)

res_at_V!(F, u, V) = residual_stage34!(F, u, V, m.cl, m.gm, m.c_eq, m.ly, false)

# ── Bootstrap step 1 (§11 Stage 3): gas-only with all CL sources zeroed.
#    Uniform inlet composition is the exact solution, so this is a check that
#    also pays the JIT cost of the whole SM chain cheaply.
u0 = u0_stage(m.ly, m.c_eq)
Fg = zeros(m.ly.N_TOT); zr = zeros(m.cl.N)
gas_residual!(Fg, u0, m.ly, m.gm, zr, zr, zr, false)
gas_rows = (5*m.cl.N+1):(5*m.cl.N+4*m.ly.Ng)
@printf("gas-only bootstrap: max |F_gas| at inlet composition = %.3e (expect ~0)\n\n",
        maximum(abs, Fg[gas_rows]))

t0 = time()
Vs, Us, rows, plu = run_continuation(u0, copy(V_RHE_grid), trues(89), res_at_V!;
        blocks = weight_blocks(m.ly), n_log = 5*m.cl.N,
        cache_dir = CACHE_DIR, prefix = PREFIX, verbose = false)
@printf("wall time: %.1f s\n", time() - t0)
export_all("stage3", Vs, Us, m.ly, m.cl, m.gm, rows)

@printf("\n=== Stage 3 summary ===\n")
@printf("converged %d/%d, mean iters = %.2f, max ||F||inf = %.3e\n",
        length(Vs), 89, sum(r.iters for r in rows)/length(rows),
        maximum(r.Finf for r in rows))

# ── §11 Stage 3 physicality checks
println("\n-- gas composition (mole fractions) --")
for V in (-0.300, -1.200, -1.650, -2.500)
    k = argmin(abs.(Vs .- V)); u = Us[k]
    wI, pI = cell_gas(u, m.ly, 1)                       # GDL inlet cell
    wC, pC = cell_gas(u, m.ly, m.ly.N_GDL + m.cl.N)     # last CL cell
    xI = omega_to_x(wI, Mn_from_omega(wI))
    xC = omega_to_x(wC, Mn_from_omega(wC))
    @printf("V_RHE=%.3f  GDL[1]: w_CO2=%.4f x=[%.4f %.4f %.4f %.4f] | CL[end]: w_CO2=%.4f x=[%.4f %.4f %.4f %.4f]\n",
            Vs[k], wI[1], xI[1], xI[2], xI[3], xI[4], wC[1], xC[1], xC[2], xC[3], xC[4])
end

println("\n-- Henry consistency at the GDL/CL boundary (CL cell 1) --")
for V in (-1.200, -2.500)
    k = argmin(abs.(Vs .- V)); u = Us[k]
    c = unpack_log_conc(u, m.cl.N)
    w1, p1 = cell_gas(u, m.ly, m.ly.N_GDL + 1)
    H = sechenov_H_CO2(@view c[:,1]); pc = pCO2_atm(w1, p1)
    @printf("V_RHE=%.3f  p_CO2=%.4f atm, H_eff=%.3f → c_CO2^eq=%.4f vs c_CO2(aq)[1]=%.4f mol/m³\n",
            Vs[k], pc, H, H*pc, c[5,1])
end

println("\n-- pressure pin check (Stage 3: p must be exactly uniform) --")
k = argmin(abs.(Vs .- (-2.500))); u = Us[k]
ps = [cell_gas(u, m.ly, g)[2] for g in 1:m.ly.Ng]
@printf("max |p − p_total| = %.3e Pa\n", maximum(abs.(ps .- p_total)))

println("\n-- Stage 2 vs Stage 3 currents --")
for V in (-0.816, -1.205, -1.644, -2.079, -2.413)
    local k = argmin(abs.(Vs .- V))
    u3 = Us[k]; u2 = load_solution(joinpath(@__DIR__,"output_stage2","cache"), "s2_", Vs[k])
    V_SHE = Vs[k] - V_RHE_to_SHE_offset
    a = compute_currents(u3, V_SHE, m.cl); b = compute_currents(u2, V_SHE, m.cl)
    @printf("V_RHE=%.3f  S2: i_CO=%8.2f i_H2=%8.2f | S3: i_CO=%8.2f i_H2=%8.2f  mA/cm²\n",
            Vs[k], b[1]/10, b[2]/10, a[1]/10, a[2]/10)
end

println("\n-- CO2 mass balance (§12.6) --")
let
    for Vt in (-1.200, -1.650, -2.500)
        k = argmin(abs.(Vs .- Vt)); u = Us[k]
        c = unpack_log_conc(u, m.cl.N)
        w1, p1 = cell_gas(u, m.ly, m.ly.N_GDL + 1)
        J_in  = diffusion_flux(sechenov_H_CO2(@view c[:,1])*pCO2_atm(w1,p1), c[5,1],
                               D_eff[5], m.cl.dxc[1])
        J_out = -k_MT[5]*(m.c_eq[5] - c[5,end])
        r1,_ = rate_profiles(u, m.ly, m.cl, Vs[k]-V_RHE_to_SHE_offset)
        intRPT = 0.0; intRCO = 0.0; intbuf = 0.0
        for ix in 1:m.cl.N
            wg,pg = cell_gas(u, m.ly, m.ly.N_GDL+ix)
            intRPT += a_v_GL*k_GL()*(sechenov_H_CO2(@view c[:,ix])*pCO2_atm(wg,pg) - c[5,ix])*m.cl.dx[ix]
            intRCO += r1[ix]*m.cl.dx[ix]
            intbuf += eps_I*buffer_sources(@view c[:,ix])[5]*m.cl.dx[ix]
        end
        bal = J_in - J_out + intRPT - intRCO + intbuf
        @printf("V_RHE=%.3f  CO2(aq) balance = %+.3e mol/m²/s  (scale ∫R_PT = %.3e)\n",
                Vs[k], bal, abs(intRPT))
    end
end

println("\nSTAGE 3 COMPLETE.")
