# export_data.jl — CSV export of everything plot_results.py needs.

const DATA_DIR = joinpath(@__DIR__, "data")

function export_polarization(tag::String, Vs, Us, ly::Layout, cl::Mesh1D, rows)
    mkpath(DATA_DIR)
    open(joinpath(DATA_DIR, "$(tag)_polarization.csv"), "w") do io
        println(io, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO,FE_H2,iters,Finf")
        for (k, V) in enumerate(Vs)
            u = Us[k]
            V_SHE = V - V_RHE_to_SHE_offset
            iCO, iH2 = ly.stage >= 5 ? compute_currents5(u, ly, cl) :
                                       compute_currents(u, V_SHE, cl)
            iCO /= 10.0; iH2 /= 10.0                    # A/m² → mA/cm²
            it = iCO + iH2
            @printf(io, "%.6f,%.6f,%.8e,%.8e,%.8e,%.6f,%.6f,%d,%.4e\n",
                    V, V_SHE, iCO, iH2, it,
                    it > 0 ? 100*iCO/it : 0.0, it > 0 ? 100*iH2/it : 0.0,
                    rows[k].iters, rows[k].Finf)
        end
    end
end

function export_liquid(tag::String, Vs, Us, ly::Layout, cl::Mesh1D)
    mkpath(DATA_DIR)
    open(joinpath(DATA_DIR, "$(tag)_liquid.csv"), "w") do io
        println(io, "V_RHE,x_um,pH,cCO2,cOH,cHCO3,cCO3,cCs,cH,R_CO2R,R_HER")
        for (k, V) in enumerate(Vs)
            u = Us[k]
            V_SHE = V - V_RHE_to_SHE_offset
            c = unpack_log_conc(u, cl.N)
            r1, r2 = rate_profiles(u, ly, cl, V_SHE)
            for ix in 1:cl.N
                cCs = c[2,ix] + c[3,ix] + 2*c[4,ix] - c[1,ix] - C_fix*eps_I
                @printf(io, "%.6f,%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        V, cl.x_centers[ix]*1e6, -log10(max(c[1,ix]/1000, 1e-30)),
                        c[5,ix], c[2,ix], c[3,ix], c[4,ix], cCs, c[1,ix], r1[ix], r2[ix])
            end
        end
    end
end

function export_gas(tag::String, Vs, Us, ly::Layout, gm::GasMesh)
    ly.n_gas == 0 && return
    mkpath(DATA_DIR)
    open(joinpath(DATA_DIR, "$(tag)_gas.csv"), "w") do io
        println(io, "V_RHE,x_um,p_CO2_atm,p_CO_atm,p_H2_atm,p_H2O_atm,p_tot_Pa")
        for (k, V) in enumerate(Vs)
            u = Us[k]
            for g in 1:gm.N
                w, p = cell_gas(u, ly, g)
                Mn = Mn_from_omega(w)
                x  = omega_to_x(w, Mn)
                @printf(io, "%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        V, gm.x_centers[g]*1e6,
                        x[1]*p/101325, x[2]*p/101325, x[3]*p/101325, x[4]*p/101325, p)
            end
        end
    end
end

function export_potentials(tag::String, Vs, Us, ly::Layout, gm::GasMesh, cl::Mesh1D)
    ly.stage >= 5 || return
    mkpath(DATA_DIR)
    open(joinpath(DATA_DIR, "$(tag)_phis.csv"), "w") do io
        println(io, "V_RHE,x_um,phi_s")
        for (k, V) in enumerate(Vs), g in 1:gm.N
            @printf(io, "%.6f,%.6f,%.8e\n", V, gm.x_centers[g]*1e6, Us[k][phi_s_idx(ly,g)])
        end
    end
    open(joinpath(DATA_DIR, "$(tag)_phiL.csv"), "w") do io
        println(io, "V_RHE,x_um,x_norm,phi_L")
        for (k, V) in enumerate(Vs), ix in 1:cl.N
            @printf(io, "%.6f,%.6f,%.6f,%.8e\n", V, cl.x_centers[ix]*1e6,
                    cl.x_centers[ix]/L_CL, Us[k][phi_L_idx(ly,ix)])
        end
    end
end

function export_all(tag, Vs, Us, ly, cl, gm, rows)
    export_polarization(tag, Vs, Us, ly, cl, rows)
    export_liquid(tag, Vs, Us, ly, cl)
    export_gas(tag, Vs, Us, ly, gm)
    export_potentials(tag, Vs, Us, ly, gm, cl)
    @printf("exported CSVs for %s (%d voltages)\n", tag, length(Vs))
end
