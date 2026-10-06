# plots.jl — required diagnostic plots (§12.7). Uses Plots.jl (GR backend) to keep
# everything in-process (build choice; guide §13 suggested matplotlib/Python).
using Plots

# ── A1: 3-panel polarization summary (linear i, log i, FE) vs −E_cath [V vs SHE]
# model_V_RHE, model_iCO, model_iH2 in mA/cm². Experimental points from §16.1.
function plot_A1(outdir, model_V_RHE, model_iCO, model_iH2; title_prefix="Stage")
    negE = .-(model_V_RHE .- V_RHE_to_SHE_offset)      # −E_cath [V vs SHE]
    iTot = model_iCO .+ model_iH2
    negE_exp = .-E_exp_SHE
    FE_CO = model_iCO ./ max.(iTot, 1e-30) .* 100
    FE_H2 = model_iH2 ./ max.(iTot, 1e-30) .* 100

    p1 = plot(negE, model_iCO, lw=2, c=:blue, label="CO model",
              xlabel="−E_cath [V vs SHE]", ylabel="mA/cm²", title="(a) Linear polarization")
    plot!(p1, negE, model_iH2, lw=2, c=:red, label="H₂ model")
    plot!(p1, negE, iTot, lw=2, c=:black, label="total model")
    scatter!(p1, negE_exp, i_CO_exp, m=:square, mc=:blue, ms=5, label="CO exp")
    scatter!(p1, negE_exp, i_H2_exp, m=:utriangle, mc=:red, ms=5, label="H₂ exp")

    p2 = plot(negE, max.(model_iCO,1e-3), lw=2, c=:blue, yscale=:log10, label="CO model",
              xlabel="−E_cath [V vs SHE]", ylabel="mA/cm² (log)", title="(b) Log polarization")
    plot!(p2, negE, max.(model_iH2,1e-3), lw=2, c=:red, label="H₂ model")
    plot!(p2, negE, max.(iTot,1e-3), lw=2, c=:black, label="total model")
    scatter!(p2, negE_exp, i_CO_exp, m=:square, mc=:blue, ms=5, label="CO exp")
    scatter!(p2, negE_exp, i_H2_exp, m=:utriangle, mc=:red, ms=5, label="H₂ exp")

    p3 = plot(negE, FE_CO, lw=2, c=:blue, label="FE_CO model", ylims=(0,100),
              xlabel="−E_cath [V vs SHE]", ylabel="%", title="(c) Faradaic efficiency")
    plot!(p3, negE, FE_H2, lw=2, c=:red, label="FE_H₂ model")
    scatter!(p3, negE_exp, FE_CO_exp, m=:square, mc=:blue, ms=5, label="FE_CO exp")
    scatter!(p3, negE_exp, FE_H2_exp, m=:utriangle, mc=:red, ms=5, label="FE_H₂ exp")

    fig = plot(p1, p2, p3, layout=(1,3), size=(1500,450), legend=:best,
               plot_title="$title_prefix — A1 polarization summary")
    savefig(fig, joinpath(outdir, "A1_polarization.png"))
    return fig
end

# ── Overlay polarization across stages. stage_series: Vector of (label, Vr, iCO, iH2).
function plot_overlay(outdir, stage_series; fname="overlay_polarization.png", title="Polarization overlay")
    negE_exp = .-E_exp_SHE
    styles = [:solid, :dash, :dashdot, :dot, :solid, :dash]
    p1 = plot(xlabel="−E_cath [V vs SHE]", ylabel="mA/cm²", title="(a) Linear")
    p2 = plot(xlabel="−E_cath [V vs SHE]", ylabel="mA/cm² (log)", yscale=:log10, title="(b) Log")
    for (k,(lab,Vr,iCO,iH2)) in enumerate(stage_series)
        negE = .-(Vr .- V_RHE_to_SHE_offset); ls = styles[mod1(k,length(styles))]
        plot!(p1, negE, iCO, lw=2, c=:blue, ls=ls, label="$lab CO")
        plot!(p1, negE, iH2, lw=2, c=:red,  ls=ls, label="$lab H₂")
        plot!(p2, negE, max.(iCO,1e-3), lw=2, c=:blue, ls=ls, label="$lab CO")
        plot!(p2, negE, max.(iH2,1e-3), lw=2, c=:red,  ls=ls, label="$lab H₂")
    end
    scatter!(p1, negE_exp, i_CO_exp, m=:square, mc=:blue, ms=5, label="CO exp")
    scatter!(p1, negE_exp, i_H2_exp, m=:utriangle, mc=:red, ms=5, label="H₂ exp")
    scatter!(p2, negE_exp, i_CO_exp, m=:square, mc=:blue, ms=5, label="CO exp")
    scatter!(p2, negE_exp, i_H2_exp, m=:utriangle, mc=:red, ms=5, label="H₂ exp")
    fig = plot(p1, p2, layout=(1,2), size=(1200,500), legend=:best, plot_title=title)
    savefig(fig, joinpath(outdir, fname))
    return fig
end

# ── B1: 2×2 gas-phase diagnostic (Stage 3+). gas_profiles: Vector of (V_RHE, pCO2,pCO,pH2,ptot),
# each a length-(N_GDL+N_CL) vector; x_gas_μm is the combined GDL+CL cell-center coordinate [μm].
function plot_B1(outdir, x_gas_μm, gas_series; title_prefix="Stage")
    nV = length(gas_series)
    zvals = [-(g[1] - V_RHE_to_SHE_offset) for g in gas_series]
    zmin, zmax = minimum(zvals), maximum(zvals)
    labels = ["(a) p_CO₂ [atm]","(b) p_CO [atm]","(c) p_H₂ [atm]","(d) p_total [atm]"]
    ps = [plot(xlabel="x [μm]", title=labels[k], legend=false) for k in 1:4]
    for (j,g) in enumerate(gas_series)
        z = zvals[j]
        for (k, ser) in enumerate((g[2], g[3], g[4], g[5]))
            plot!(ps[k], x_gas_μm, ser; line_z=z, clims=(zmin,zmax), c=:viridis, colorbar=false, lw=1.3)
        end
    end
    for k in 1:4
        vline!(ps[k], [0.0], c=:gray, ls=:dash, label="")   # GDL/CL interface at x=0
    end
    yv = collect(range(zmin, zmax, length=200))
    cb = heatmap([0.0], yv, reshape(yv, :, 1); c=:viridis, legend=false, colorbar=false,
                 xticks=false, ymirror=true, ylabel="−E_cath [V vs SHE]", title="cathodic →", titlefontsize=8)
    l = @layout [grid(2,2) cbar{0.07w}]
    fig = plot(ps..., cb, layout=l, size=(1200,900),
               plot_title="$title_prefix — B1 gas-phase ($nV voltages; color = −E_cath; dashed = GDL/CL)")
    savefig(fig, joinpath(outdir, "B1_gas.png"))
    return fig
end

# ── D1/D2: φ_s (GDL+CL) and φ_L (CL) profiles. series: (V_RHE, x_s_μm, φs, x_L_μm, φL).
function plot_D(outdir, series; title_prefix="Stage 5")
    nV = length(series)
    zvals = [-(s[1] - V_RHE_to_SHE_offset) for s in series]
    zmin, zmax = minimum(zvals), maximum(zvals)
    p1 = plot(xlabel="x [μm]  (−L_GDL → L_CL)", ylabel="φ_s [V vs SHE]",
              title="D1: solid potential φ_s (GDL+CL)", legend=false)
    p2 = plot(xlabel="x [μm] (CL)", ylabel="φ_L [V vs SHE]",
              title="D2: liquid potential φ_L (CL)", legend=false)
    for (j,s) in enumerate(series)
        z = zvals[j]
        plot!(p1, s[2], s[3]; line_z=z, clims=(zmin,zmax), c=:viridis, colorbar=false, lw=1.3)
        plot!(p2, s[4], s[5]; line_z=z, clims=(zmin,zmax), c=:viridis, colorbar=false, lw=1.3)
    end
    vline!(p1, [0.0], c=:gray, ls=:dash)
    yv = collect(range(zmin, zmax, length=200))
    cb = heatmap([0.0], yv, reshape(yv,:,1); c=:viridis, legend=false, colorbar=false,
                 xticks=false, ymirror=true, ylabel="−E_cath [V vs SHE]", title="cathodic →", titlefontsize=8)
    l = @layout [grid(1,2) cbar{0.06w}]
    fig = plot(p1, p2, cb, layout=l, size=(1350,520),
               plot_title="$title_prefix — potentials ($nV voltages; color = −E_cath)")
    savefig(fig, joinpath(outdir, "D_potentials.png"))
    return fig
end

# ── Stage 4 Darcy diagnostic: pressure drop (p − p_total) and Darcy velocity vs x.
# series: Vector of (V_RHE, dp_Pa[n], uG_ms[n]) — dp per cell, u_G per cell-face-avg.
function plot_darcy(outdir, x_gas_μm, series; title_prefix="Stage 4")
    nV = length(series)
    zvals = [-(s[1] - V_RHE_to_SHE_offset) for s in series]
    zmin, zmax = minimum(zvals), maximum(zvals)
    p1 = plot(xlabel="x [μm]", ylabel="p − p_total [Pa]", title="(a) Darcy pressure drop", legend=false)
    p2 = plot(xlabel="x [μm]", ylabel="u_G [m/s]", title="(b) Darcy velocity", legend=false)
    for (j,s) in enumerate(series)
        z = zvals[j]
        plot!(p1, x_gas_μm, s[2]; line_z=z, clims=(zmin,zmax), c=:viridis, colorbar=false, lw=1.3)
        plot!(p2, x_gas_μm, s[3]; line_z=z, clims=(zmin,zmax), c=:viridis, colorbar=false, lw=1.3)
    end
    vline!(p1, [0.0], c=:gray, ls=:dash); vline!(p2, [0.0], c=:gray, ls=:dash)
    yv = collect(range(zmin, zmax, length=200))
    cb = heatmap([0.0], yv, reshape(yv,:,1); c=:viridis, legend=false, colorbar=false,
                 xticks=false, ymirror=true, ylabel="−E_cath [V vs SHE]", title="cathodic →", titlefontsize=8)
    l = @layout [grid(1,2) cbar{0.06w}]
    fig = plot(p1, p2, cb, layout=l, size=(1350,500),
               plot_title="$title_prefix — Darcy pressure & velocity (dashed = GDL/CL)")
    savefig(fig, joinpath(outdir, "darcy_pressure_velocity.png"))
    return fig
end

# ── A2: 3×3 liquid-phase diagnostic. profiles is a Vector of (V_RHE, c[5×N], R_CO2R[N], R_HER[N]).
# Curves colored continuously by −E_cath [V vs SHE] (viridis: dark=mild → light=cathodic),
# with a single shared colorbar strip in a 10th panel.
function plot_A2(outdir, cl_mesh, profiles; title_prefix="Stage")
    xμm = cl_mesh.xc .* 1e6
    nV = length(profiles)
    zvals = [-(pr[1] - V_RHE_to_SHE_offset) for pr in profiles]   # −E_cath [V vs SHE]
    zmin, zmax = minimum(zvals), maximum(zvals)
    labels = ["(a) pH","(b) CO₂(aq) [mol/m³]","(c) OH⁻ [mol/m³]",
              "(d) HCO₃⁻ [mol/m³]","(e) CO₃²⁻ [mol/m³]","(f) Cs⁺ [mol/m³]",
              "(g) H⁺ [mol/m³]","(h) R_CO2R [mol/m³/s]","(i) R_HER [mol/m³/s]"]
    ps = [plot(xlabel="x [μm]", title=labels[k], legend=false) for k in 1:9]
    for (j, pr) in enumerate(profiles)
        V, c, rco2r, rher = pr
        z = zvals[j]
        cCs = [c[2,ix]+c[3,ix]+2*c[4,ix]-c[1,ix]-C_fix*eps_I for ix in 1:cl_mesh.N]
        pH  = [-log10(max(c[1,ix]/1000,1e-20)) for ix in 1:cl_mesh.N]
        series = (pH, c[5,:], c[2,:], c[3,:], c[4,:], cCs, c[1,:], rco2r, rher)
        for k in 1:9
            plot!(ps[k], xμm, series[k]; line_z=z, clims=(zmin,zmax),
                  c=:viridis, colorbar=false, lw=1.3)
        end
    end
    # Shared colorbar strip (10th panel): vertical viridis gradient labeled by −E_cath.
    yv = collect(range(zmin, zmax, length=200))
    cb = heatmap([0.0], yv, reshape(yv, :, 1); c=:viridis, legend=false, colorbar=false,
                 xticks=false, ymirror=true, ylabel="−E_cath [V vs SHE]",
                 title="cathodic →", titlefontsize=8)
    l = @layout [grid(3,3) cbar{0.07w}]
    fig = plot(ps..., cb, layout=l, size=(1550,1100),
               plot_title="$title_prefix — A2 liquid-phase ($nV voltages; color = −E_cath)")
    savefig(fig, joinpath(outdir, "A2_liquid.png"))
    return fig
end
