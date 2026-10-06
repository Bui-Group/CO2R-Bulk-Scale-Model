# checks.jl — the §11/§12 physicality checks, run automatically at each stop gate

using Printf

"""Count sign changes in the first difference of `y`."""
function sign_changes(y::Vector{Float64})
    d = diff(y)
    d = d[abs.(d) .> 1e-12 * max(maximum(abs, y), 1e-30)]
    isempty(d) && return 0
    s = sign.(d)
    return count(i -> s[i] != s[i+1], 1:length(s)-1)
end

function report_polarization(Vs, Us, ly, dm)
    iCO = Float64[]; iH2 = Float64[]
    for (k,V) in enumerate(Vs)
        a,b,_ = compute_currents(Us[k], ly, dm, V)
        push!(iCO, a/10); push!(iH2, b/10)     # mA/cm²
    end
    itot = iCO .+ iH2
    FE = 100 .* iCO ./ max.(itot, 1e-30)
    kpk = argmax(iCO); kfe = argmax(FE)
    @printf("  i_CO   : min %.4g  max %.4g mA/cm^2 (peak at V_RHE=%.3f)\n",
            minimum(iCO), maximum(iCO), Vs[kpk])
    @printf("  i_H2   : min %.4g  max %.4g mA/cm^2\n", minimum(iH2), maximum(iH2))
    @printf("  i_tot  : min %.4g  max %.4g mA/cm^2\n", minimum(itot), maximum(itot))
    @printf("  FE_CO  : peak %.1f %% at V_RHE=%.3f\n", FE[kfe], Vs[kfe])
    sc_co = sign_changes(iCO); sc_h2 = sign_changes(iH2)
    @printf("  sign changes in di_CO/dV = %d (target 1), di_H2/dV = %d (target 0)\n",
            sc_co, sc_h2)
    mono_h2 = all(diff(iH2) .> 0)
    @printf("  i_H2 strictly monotone increasing with |V|: %s\n", mono_h2)
    return (iCO=iCO, iH2=iH2, itot=itot, FE=FE, sc_co=sc_co, sc_h2=sc_h2, mono_h2=mono_h2)
end

function compare_to_experiment(Vs, Us, ly, dm)
    println("  Experimental comparison (interpolated onto the model sweep):")
    iCO = Float64[]; iH2 = Float64[]
    for (k,V) in enumerate(Vs)
        a,b,_ = compute_currents(Us[k], ly, dm, V); push!(iCO,a/10); push!(iH2,b/10)
    end
    ord = sortperm(Vs)
    Vs_s = Vs[ord]; iCO_s = iCO[ord]; iH2_s = iH2[ord]
    lin(xq, x, y) = begin
        j = searchsortedfirst(x, xq)
        j <= 1 && return y[1]
        j > length(x) && return y[end]
        t = (xq - x[j-1])/(x[j]-x[j-1]); y[j-1] + t*(y[j]-y[j-1])
    end
    @printf("   %8s %10s %10s %8s | %10s %10s %8s\n",
            "V_RHE","iCO_mod","iCO_exp","err%","iH2_mod","iH2_exp","err%")
    for (k,V) in enumerate(E_exp_RHE)
        mc = lin(V, Vs_s, iCO_s); mh = lin(V, Vs_s, iH2_s)
        @printf("   %8.3f %10.3f %10.3f %8.1f | %10.3f %10.3f %8.1f\n",
                V, mc, i_CO_exp[k], 100*(mc-i_CO_exp[k])/i_CO_exp[k],
                mh, i_H2_exp[k], 100*(mh-i_H2_exp[k])/i_H2_exp[k])
    end
end

"""Liquid-phase profile checks (§11 Stage 1/2, §12.3)."""
function report_liquid(Vs, Us, ly, dm)
    kend = argmin(Vs)            # most cathodic
    V = Vs[kend]; u = Us[kend]
    V_SHE = V - V_RHE_to_SHE_offset
    Rco, Rher, c = compute_rates(u, ly, V_SHE)
    pH = [-log10(max(c[1,ix]/1000, 1e-30)) for ix in 1:ly.N_CL]
    cCs = [cCs_from_electroneutrality(c[1,ix],c[2,ix],c[3,ix],c[4,ix]) for ix in 1:ly.N_CL]
    @printf("  At the most cathodic voltage V_RHE = %.3f:\n", V)
    @printf("    pH:        x=0 -> %.2f ; x=L_CL -> %.2f  (should decrease toward EL)\n",
            pH[1], pH[end])
    @printf("    c_CO2(aq): x=0 -> %.4g ; x=L_CL -> %.4g mol/m3\n", c[5,1], c[5,end])
    @printf("    c_OH:      x=0 -> %.4g ; x=L_CL -> %.4g mol/m3 (bulk eq %.3e)\n",
            c[2,1], c[2,end], dm.c_eq[2])
    @printf("    c_HCO3:    x=0 -> %.4g ; x=L_CL -> %.4g mol/m3\n", c[3,1], c[3,end])
    @printf("    c_CO3:     x=0 -> %.4g ; x=L_CL -> %.4g mol/m3\n", c[4,1], c[4,end])
    @printf("    c_Cs:      x=0 -> %.4g ; x=L_CL -> %.4g mol/m3\n", cCs[1], cCs[end])
    @printf("    max R_CO2R = %.4g, max R_HER = %.4g mol/m3/s\n",
            maximum(Rco), maximum(Rher))
    @printf("    electroneutrality residual (by construction) = 0 (c_Cs is the closure)\n")
    # pH range over the whole sweep
    pHmin = Inf; pHmax = -Inf
    for (k,Vk) in enumerate(Vs)
        ck = unpack_log_conc(Us[k], ly.N_CL)
        for ix in 1:ly.N_CL
            p = -log10(max(ck[1,ix]/1000, 1e-30))
            pHmin = min(pHmin,p); pHmax = max(pHmax,p)
        end
    end
    @printf("    pH range across the whole sweep: %.2f – %.2f\n", pHmin, pHmax)
end

"""CO₂ mass balance across the CL (§12.6)."""
function co2_mass_balance(u, ly, dm, V)
    V_SHE = V - V_RHE_to_SHE_offset
    Rco, Rher, c = compute_rates(u, ly, V_SHE)
    N_CL = ly.N_CL
    phi_L = ly.stage >= 5 ? [u[phi_L_idx(ly,ix)] for ix in 1:N_CL] : zeros(N_CL)
    # left face
    if ly.stage <= 2
        cbc = c_CO2_bc_s12
    else
        w = gas_omega(u, ly, ly.N_GDL+1); p = gas_p(u, ly, ly.N_GDL+1)
        x = omega_to_x(w[1],w[2],w[3],w[4])
        cbc = sechenov_H_CO2(view(c,:,1)) * x[1]*p/101325
    end
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    Jin = ly.stage >= 5 ? 0.0 : diffusion_flux(cbc, c[5,1], D_eff[5], dm.cl.dx_cc[1])
    if ly.stage == 1
        Jout = diffusion_flux(c[5,N_CL], dm.c_eq[5], D_eff[5], dm.cl.dx_cc[N_CL+1])
    elseif ly.stage <= 4
        Jout = -k_MT[5]*(dm.c_eq[5] - c[5,N_CL])
    else
        Jout = np_flux(c[5,N_CL], dm.c_eq[5], phi_L[N_CL], 0.0,
                       D_eff[5], z_sp[5], D_eff[5]/k_MT[5])
    end
    src = 0.0
    for ix in 1:N_CL
        buf = buffer_sources(view(c,:,ix))
        s = eps_I*buf[5] - Rco[ix]
        if ly.stage >= 3
            w = gas_omega(u, ly, ly.N_GDL+ix); p = gas_p(u, ly, ly.N_GDL+ix)
            x = omega_to_x(w[1],w[2],w[3],w[4])
            s += a_v_GL*k_GL()*(sechenov_H_CO2(view(c,:,ix))*x[1]*p/101325 - c[5,ix])
        end
        src += s*dm.cl.dx[ix]
    end
    return Jin - Jout + src
end
