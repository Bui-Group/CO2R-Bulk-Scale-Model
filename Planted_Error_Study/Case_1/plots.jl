# plots.jl — Required diagnostic plots (§12.7). Julia/Plots.jl (GR, headless).
# A1: 3-panel polarization summary.  A2: 3×3 liquid-phase profiles.
# All voltage x-axes use −E_cath [V vs SHE] (positive = more cathodic).

ENV["GKSwstype"] = "100"   # headless GR
using Plots
gr()

"Recover Cs⁺ from electroneutrality (§2): c_Cs = c_OH + c_HCO3 + 2c_CO3 − c_H − C_fix·ε_I."
cs_from_eln(c) = c[2] + c[3] + 2*c[4] - c[1] - C_fix*eps_I

"""
    plot_B1(results, cl_mesh, gdl_mesh, outpath; stage_label)

2×2 gas-phase diagnostic across GDL+CL at all sweep voltages (color = V_RHE):
p_CO₂, p_CO, p_H₂ [atm] and p_total [Pa]. GDL/CL interface marked at x=0.
"""
function plot_B1(results, cl_mesh, gdl_mesh, outpath; stage_label="Stage 3")
    NG = gdl_mesh.N; NC = cl_mesh.N
    xμm = vcat(gdl_mesh.x_centers, cl_mesh.x_centers) .* 1e6
    Vvals = [r[1] for r in results]; Vmin, Vmax = minimum(Vvals), maximum(Vvals)
    panels = [plot() for _ in 1:4]
    titles = ["(a) p_CO₂ [atm]", "(b) p_CO [atm]", "(c) p_H₂ [atm]", "(d) p_total [Pa]"]
    for (V, u) in results
        zc = fill(V, NG + NC)
        pCO2 = zeros(NG+NC); pCO = zeros(NG+NC); pH2 = zeros(NG+NC); ptot = zeros(NG+NC)
        for g in 1:(NG+NC)
            w, p = gas_state(u, g, NC, NG)
            Mn = Mn_from_omega(w); x = omega_to_x(w, Mn)
            pCO2[g] = x[1]*p/101325; pCO[g] = x[2]*p/101325; pH2[g] = x[3]*p/101325; ptot[g] = p
        end
        for (pl, y) in zip(panels, (pCO2, pCO, pH2, ptot))
            plot!(pl, xμm, y; line_z=zc, clims=(Vmin,Vmax), c=:viridis, colorbar=false, label="", lw=1.0)
        end
    end
    for (pl, t) in zip(panels, titles)
        plot!(pl, title=t, xlabel="x [μm]  (GDL<0, CL>0)", titlefontsize=9)
        vline!(pl, [0.0], ls=:dash, color=:gray, label="")
    end
    yv = collect(range(Vmin, Vmax, length=128))
    cb = heatmap([1.0], yv, reshape(yv, :, 1); c=:viridis, colorbar=false, xticks=false,
                 ylabel="V_RHE [V]", title="color scale", titlefontsize=9, framestyle=:box)
    lay = @layout [grid(2,2) cbar{0.07w}]
    fig = plot(panels..., cb, layout=lay, size=(1300,950),
               plot_title="$stage_label — B1 gas-phase (color = V_RHE)")
    savefig(fig, outpath)
    return outpath
end

"""
    plot_A1(results, cl_mesh, outpath; stage_label)

results :: Vector of (V_RHE, u). 3-panel polarization summary with Lees overlay.
"""
function plot_A1(results, cl_mesh, outpath; stage_label="Stage 1")
    Vrhe = [r[1] for r in results]
    negVSHE = -(Vrhe .- V_RHE_to_SHE_offset)        # −E_cath [V vs SHE]
    iCO = Float64[]; iH2 = Float64[]; itot = Float64[]
    for (V, u) in results
        Vsh = V - V_RHE_to_SHE_offset
        co = compute_i_CO(u, Vsh, cl_mesh) / 10.0    # A/m² → mA/cm²
        h2 = compute_i_H2(u, Vsh, cl_mesh) / 10.0
        push!(iCO, co); push!(iH2, h2); push!(itot, co + h2)
    end
    negVSHE_exp = -E_exp_SHE

    p_lin = plot(negVSHE, iCO, label="i_CO (model)", lw=2, color=:teal,
                 xlabel="−E_cath [V vs SHE]", ylabel="i [mA/cm²]", title="(a) Linear")
    plot!(p_lin, negVSHE, iH2, label="i_H₂ (model)", lw=2, color=:firebrick)
    plot!(p_lin, negVSHE, itot, label="i_total (model)", lw=2, color=:black, ls=:dash)
    scatter!(p_lin, negVSHE_exp, i_CO_exp, label="i_CO (exp)", marker=:square, ms=5, color=:teal)
    scatter!(p_lin, negVSHE_exp, i_H2_exp, label="i_H₂ (exp)", marker=:utriangle, ms=5, color=:firebrick)

    safe(x) = max.(x, 1e-3)
    p_log = plot(negVSHE, safe(iCO), label="i_CO", lw=2, color=:teal, yscale=:log10,
                 xlabel="−E_cath [V vs SHE]", ylabel="i [mA/cm²] (log)", title="(b) Log")
    plot!(p_log, negVSHE, safe(iH2), label="i_H₂", lw=2, color=:firebrick)
    plot!(p_log, negVSHE, safe(itot), label="i_total", lw=2, color=:black, ls=:dash)
    scatter!(p_log, negVSHE_exp, i_CO_exp, label="", marker=:square, ms=5, color=:teal)
    scatter!(p_log, negVSHE_exp, i_H2_exp, label="", marker=:utriangle, ms=5, color=:firebrick)

    feCO = 100 .* iCO ./ max.(itot, 1e-12)
    feH2 = 100 .* iH2 ./ max.(itot, 1e-12)
    p_fe = plot(negVSHE, feCO, label="FE_CO (model)", lw=2, color=:teal,
                xlabel="−E_cath [V vs SHE]", ylabel="FE [%]", title="(c) Faradaic efficiency",
                ylims=(0,100))
    plot!(p_fe, negVSHE, feH2, label="FE_H₂ (model)", lw=2, color=:firebrick)
    scatter!(p_fe, negVSHE_exp, FE_CO_exp, label="FE_CO (exp)", marker=:square, ms=5, color=:teal)
    scatter!(p_fe, negVSHE_exp, FE_H2_exp, label="FE_H₂ (exp)", marker=:utriangle, ms=5, color=:firebrick)

    fig = plot(p_lin, p_log, p_fe, layout=(1,3), size=(1500,460),
               plot_title="$stage_label — A1 polarization summary", legend=:best)
    savefig(fig, outpath)
    return outpath
end

"""
    plot_polarization_overlay(stage_results, labels, cl_mesh, outpath; title)

Overlay i_CO / i_H₂ / i_total for several stages (each `results` a Vector of
(V_RHE,u)) on linear + log panels, with Lees experimental scatter.
"""
function plot_polarization_overlay(stage_results, labels, cl_mesh, outpath; title="Overlay")
    styles = [:solid, :dash, :dot, :dashdot]
    p_lin = plot(xlabel="−E_cath [V vs SHE]", ylabel="i [mA/cm²]", title="Linear")
    p_log = plot(xlabel="−E_cath [V vs SHE]", ylabel="i [mA/cm²] (log)", title="Log", yscale=:log10)
    for (s, (results, lab)) in enumerate(zip(stage_results, labels))
        Vrhe = [r[1] for r in results]
        negV = -(Vrhe .- V_RHE_to_SHE_offset)
        iCO = Float64[]; iH2 = Float64[]
        for (V,u) in results
            Vsh = V - V_RHE_to_SHE_offset
            push!(iCO, compute_i_CO(u,Vsh,cl_mesh)/10); push!(iH2, compute_i_H2(u,Vsh,cl_mesh)/10)
        end
        itot = iCO .+ iH2; ls = styles[mod1(s,4)]
        plot!(p_lin, negV, iCO,  label="i_CO $lab",  color=:teal,      ls=ls, lw=2)
        plot!(p_lin, negV, iH2,  label="i_H₂ $lab",  color=:firebrick, ls=ls, lw=2)
        plot!(p_lin, negV, itot, label="i_tot $lab", color=:black,     ls=ls, lw=1.5)
        sf(x)=max.(x,1e-3)
        plot!(p_log, negV, sf(iCO),  label="", color=:teal,      ls=ls, lw=2)
        plot!(p_log, negV, sf(iH2),  label="", color=:firebrick, ls=ls, lw=2)
        plot!(p_log, negV, sf(itot), label="", color=:black,     ls=ls, lw=1.5)
    end
    negVexp = -E_exp_SHE
    scatter!(p_lin, negVexp, i_CO_exp, label="i_CO exp", marker=:square, ms=5, color=:teal)
    scatter!(p_lin, negVexp, i_H2_exp, label="i_H₂ exp", marker=:utriangle, ms=5, color=:firebrick)
    scatter!(p_log, negVexp, i_CO_exp, label="", marker=:square, ms=5, color=:teal)
    scatter!(p_log, negVexp, i_H2_exp, label="", marker=:utriangle, ms=5, color=:firebrick)
    fig = plot(p_lin, p_log, layout=(1,2), size=(1200,500), plot_title=title, legend=:best)
    savefig(fig, outpath)
    return outpath
end

"""
    plot_A1_s5(results, cl_mesh, N_GDL, outpath) — Stage 5 polarization (local φ_s,φ_L currents),
with the fitted Stage-4 (pre-potential) currents overlaid to show the ohmic drop.
"""
function plot_A1_s5(results, cl_mesh, N_GDL, outpath, fit_results)
    negV(rs) = -([r[1] for r in rs] .- V_RHE_to_SHE_offset)
    s5co = Float64[]; s5h2 = Float64[]
    for (V,u) in results
        co,h2 = s5_currents(u, cl_mesh, N_GDL); push!(s5co, co/10); push!(s5h2, h2/10)
    end
    nV5 = negV(results); s5tot = s5co .+ s5h2
    f4co = Float64[]; f4h2 = Float64[]
    for (V,u) in fit_results
        push!(f4co, compute_i_CO(u, V-V_RHE_to_SHE_offset, cl_mesh)/10)
        push!(f4h2, compute_i_H2(u, V-V_RHE_to_SHE_offset, cl_mesh)/10)
    end
    nV4 = negV(fit_results)
    negVexp = -E_exp_SHE
    p_lin = plot(nV5, s5co, label="i_CO S5", lw=2, color=:teal, xlabel="−E_cath [V vs SHE]", ylabel="i [mA/cm²]", title="(a) Linear")
    plot!(p_lin, nV5, s5h2, label="i_H₂ S5", lw=2, color=:firebrick)
    plot!(p_lin, nV5, s5tot, label="i_tot S5", lw=1.5, color=:black, ls=:dash)
    plot!(p_lin, nV4, f4co, label="i_CO fit(S4)", lw=1, color=:teal, ls=:dot)
    plot!(p_lin, nV4, f4h2, label="i_H₂ fit(S4)", lw=1, color=:firebrick, ls=:dot)
    scatter!(p_lin, negVexp, i_CO_exp, label="i_CO exp", marker=:square, ms=5, color=:teal)
    scatter!(p_lin, negVexp, i_H2_exp, label="i_H₂ exp", marker=:utriangle, ms=5, color=:firebrick)
    sf(x)=max.(x,1e-3)
    p_log = plot(nV5, sf(s5co), label="i_CO S5", lw=2, color=:teal, yscale=:log10, xlabel="−E_cath [V vs SHE]", ylabel="i [mA/cm²] (log)", title="(b) Log")
    plot!(p_log, nV5, sf(s5h2), label="i_H₂ S5", lw=2, color=:firebrick)
    plot!(p_log, nV4, sf(f4co), label="", lw=1, color=:teal, ls=:dot)
    plot!(p_log, nV4, sf(f4h2), label="", lw=1, color=:firebrick, ls=:dot)
    scatter!(p_log, negVexp, i_CO_exp, label="", marker=:square, ms=5, color=:teal)
    scatter!(p_log, negVexp, i_H2_exp, label="", marker=:utriangle, ms=5, color=:firebrick)
    feCO = 100 .* s5co ./ max.(s5tot,1e-12)
    p_fe = plot(nV5, feCO, label="FE_CO S5", lw=2, color=:teal, xlabel="−E_cath [V vs SHE]", ylabel="FE [%]", title="(c) FE", ylims=(0,100))
    plot!(p_fe, nV5, 100 .- feCO, label="FE_H₂ S5", lw=2, color=:firebrick)
    scatter!(p_fe, negVexp, FE_CO_exp, label="FE_CO exp", marker=:square, ms=5, color=:teal)
    scatter!(p_fe, negVexp, FE_H2_exp, label="FE_H₂ exp", marker=:utriangle, ms=5, color=:firebrick)
    fig = plot(p_lin, p_log, p_fe, layout=(1,3), size=(1500,460), plot_title="Stage 5 — A1 polarization (solid=S5, dotted=fitted S4)", legend=:best)
    savefig(fig, outpath); return outpath
end

"D1: φ_s profile across GDL+CL [V vs SHE] at several voltages."
function plot_D1(results, cl_mesh, gdl_mesh, N_GDL, outpath)
    NG = gdl_mesh.N; N = cl_mesh.N
    xμm = vcat(gdl_mesh.x_centers, cl_mesh.x_centers) .* 1e6
    Vv = [r[1] for r in results]; Vmin,Vmax = minimum(Vv), maximum(Vv)
    p = plot(xlabel="x [μm]  (GDL<0, CL>0)", ylabel="φ_s [V vs SHE]", title="Stage 5 — D1 solid potential φ_s")
    for (V,u) in results
        φs = vcat([u[phi_s_gdl_idx(ig,N,NG)] for ig in 1:NG], [u[phi_s_cl_idx(ix,N,NG)] for ix in 1:N])
        plot!(p, xμm, φs; line_z=fill(V,NG+N), clims=(Vmin,Vmax), c=:viridis, colorbar=false, label="", lw=1.2)
    end
    vline!(p, [0.0], ls=:dash, color=:gray, label="")
    yv = collect(range(Vmin,Vmax,length=100))
    cb = heatmap([1.0], yv, reshape(yv,:,1); c=:viridis, colorbar=false, xticks=false, ylabel="V_RHE [V]", title="scale", titlefontsize=8, framestyle=:box)
    fig = plot(p, cb, layout=(@layout [a cbar{0.08w}]), size=(1000,600))
    savefig(fig, outpath); return outpath
end

"D2: φ_L profile across the CL [mV] at several voltages."
function plot_D2(results, cl_mesh, N_GDL, outpath)
    N = cl_mesh.N; xL = cl_mesh.x_centers ./ L_CL
    Vv = [r[1] for r in results]; Vmin,Vmax = minimum(Vv), maximum(Vv)
    p = plot(xlabel="x / L_CL", ylabel="φ_L [mV]", title="Stage 5 — D2 liquid potential φ_L (gauge φ_L=0 at electrolyte)")
    for (V,u) in results
        φL = [u[phi_L_idx(ix,N,N_GDL)]*1000 for ix in 1:N]
        plot!(p, xL, φL; line_z=fill(V,N), clims=(Vmin,Vmax), c=:viridis, colorbar=false, label="", lw=1.2)
    end
    yv = collect(range(Vmin,Vmax,length=100))
    cb = heatmap([1.0], yv, reshape(yv,:,1); c=:viridis, colorbar=false, xticks=false, ylabel="V_RHE [V]", title="scale", titlefontsize=8, framestyle=:box)
    fig = plot(p, cb, layout=(@layout [a cbar{0.08w}]), size=(1000,600))
    savefig(fig, outpath); return outpath
end

"""
    plot_A2(results, cl_mesh, outpath; stage_label)

3×3 liquid-phase diagnostic across the CL at all (converged) sweep voltages,
light→dark color ramp by voltage.
"""
function plot_A2(results, cl_mesh, outpath; stage_label="Stage 1")
    xμm = cl_mesh.x_centers .* 1e6
    Vvals = [r[1] for r in results]
    Vmin, Vmax = minimum(Vvals), maximum(Vvals)   # color ramp = V_RHE [V]
    panels = [plot() for _ in 1:9]
    titles = ["(a) pH", "(b) CO₂(aq) [mol/m³]", "(c) OH⁻ [mol/m³]",
              "(d) HCO₃⁻ [mol/m³]", "(e) CO₃²⁻ [mol/m³]", "(f) Cs⁺ [mol/m³]",
              "(g) H⁺ [mol/m³]", "(h) R_CO2R [mol/m³/s]", "(i) R_HER [mol/m³/s]"]
    for (V, u) in results
        Vsh = V - V_RHE_to_SHE_offset
        zc  = fill(V, cl_mesh.N)                  # color each line by its V_RHE
        c = unpack_log_conc(u, cl_mesh.N)         # 5×N Float64
        pH  = [-log10(c[1,ix]/1000.0) for ix in 1:cl_mesh.N]
        cs  = [cs_from_eln(@view c[:,ix]) for ix in 1:cl_mesh.N]
        rco = [R_CO2R(c[5,ix], Vsh, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:cl_mesh.N]
        rh2 = [R_HER(Vsh, 0.0, c[1,ix], a_v_CL, n_e) for ix in 1:cl_mesh.N]
        series = (pH, c[5,:], c[2,:], c[3,:], c[4,:], cs, c[1,:], rco, rh2)
        for (p, y) in zip(panels, series)
            plot!(p, xμm, y; line_z=zc, clims=(Vmin,Vmax), c=:viridis,
                  colorbar=false, label="", lw=1.2)
        end
    end
    for (p, t) in zip(panels, titles)
        plot!(p, title=t, xlabel="x [μm]", titlefontsize=9)
    end
    plot!(panels[7], yscale=:log10)
    # Single shared colorbar (slim right-hand column): a real heatmap strip of the
    # V_RHE scale, matching the line_z clims/colormap. (Avoids GKS viewport errors.)
    yv = collect(range(Vmin, Vmax, length=128))
    cb = heatmap([1.0], yv, reshape(yv, :, 1); c=:viridis, colorbar=false,
                 xticks=false, xlabel="", ylabel="V_RHE [V]",
                 title="color scale", titlefontsize=9, framestyle=:box)
    lay = @layout [grid(3,3) cbar{0.07w}]
    fig = plot(panels..., cb, layout=lay, size=(1500,1100),
               plot_title="$stage_label — A2 liquid-phase (color = V_RHE, $(round(Vmin,digits=2))→$(round(Vmax,digits=2)) V)")
    savefig(fig, outpath)
    return outpath
end

"Stage-5 A2: same as plot_A2 but R_CO2R/R_HER use the local φ_s, φ_L DOFs."
function plot_A2_s5(results, cl_mesh, N_GDL, outpath)
    N = cl_mesh.N; xμm = cl_mesh.x_centers .* 1e6
    Vv = [r[1] for r in results]; Vmin,Vmax = minimum(Vv), maximum(Vv)
    panels = [plot() for _ in 1:9]
    titles = ["(a) pH","(b) CO₂(aq) [mol/m³]","(c) OH⁻ [mol/m³]","(d) HCO₃⁻ [mol/m³]",
              "(e) CO₃²⁻ [mol/m³]","(f) Cs⁺ [mol/m³]","(g) H⁺ [mol/m³]","(h) R_CO2R [mol/m³/s]","(i) R_HER [mol/m³/s]"]
    for (V,u) in results
        zc = fill(V, N); c = unpack_log_conc(u, N)
        pH = [-log10(c[1,ix]/1000) for ix in 1:N]
        cs = [cs_from_eln(@view c[:,ix]) for ix in 1:N]
        rco = [R_CO2R(c[5,ix], u[phi_s_cl_idx(ix,N,N_GDL)], u[phi_L_idx(ix,N,N_GDL)], c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N]
        rh2 = [R_HER(u[phi_s_cl_idx(ix,N,N_GDL)], u[phi_L_idx(ix,N,N_GDL)], c[1,ix], a_v_CL, n_e) for ix in 1:N]
        series = (pH, c[5,:], c[2,:], c[3,:], c[4,:], cs, c[1,:], rco, rh2)
        for (p,y) in zip(panels, series)
            plot!(p, xμm, y; line_z=zc, clims=(Vmin,Vmax), c=:viridis, colorbar=false, label="", lw=1.1)
        end
    end
    for (p,t) in zip(panels, titles); plot!(p, title=t, xlabel="x [μm]", titlefontsize=9); end
    plot!(panels[7], yscale=:log10)
    yv = collect(range(Vmin,Vmax,length=128))
    cb = heatmap([1.0], yv, reshape(yv,:,1); c=:viridis, colorbar=false, xticks=false, ylabel="V_RHE [V]", title="scale", titlefontsize=9, framestyle=:box)
    fig = plot(panels..., cb, layout=(@layout [grid(3,3) cbar{0.07w}]), size=(1500,1100), plot_title="Stage 5 — A2 liquid-phase (local φ_s,φ_L)")
    savefig(fig, outpath); return outpath
end
