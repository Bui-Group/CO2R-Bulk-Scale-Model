# diagnostics.jl — current extraction + CSV writers for the Python plotter.
# Currents reported in mA/cm² (= A/m² / 10). Species order [H,OH,HCO3,CO3,CO2].

# Polarization row (mA/cm²). phi_s scalar (Stages 1–4) → φ_L = 0.
function polarization_row(u::Vector{Float64}, V_rhe::Float64, m::Mesh, phi_s::Float64)
    iCO = compute_i_CO(u, phi_s, m) / 10.0
    iH2 = compute_i_H2(u, phi_s, m) / 10.0
    itot = iCO + iH2
    feco = itot > 0 ? 100 * iCO / itot : 0.0
    return (V_rhe, V_rhe - V_RHE_to_SHE_offset, iCO, iH2, itot, feco, 100 - feco)
end

# Stage-5 polarization row using local φ_s, φ_L vectors.
function polarization_row_local(u::Vector{Float64}, V_rhe::Float64, m::Mesh,
                                phi_s::Vector{Float64}, phi_L::Vector{Float64})
    iCO = compute_i_CO_local(u, phi_s, phi_L, m) / 10.0
    iH2 = compute_i_H2_local(u, phi_s, phi_L, m) / 10.0
    itot = iCO + iH2
    feco = itot > 0 ? 100 * iCO / itot : 0.0
    return (V_rhe, V_rhe - V_RHE_to_SHE_offset, iCO, iH2, itot, feco, 100 - feco)
end

function write_polarization(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "V_rhe,V_she,i_CO,i_H2,i_tot,FE_CO,FE_H2")
        for r in rows
            @printf(io, "%.6f,%.6f,%.6e,%.6e,%.6e,%.4f,%.4f\n", r...)
        end
    end
end

# Liquid-phase profiles across the CL. `samples` = Vector of (V_rhe, u, phi_s_vec, phi_L_vec).
function write_liquid_profiles(path::AbstractString, m::Mesh, samples)
    open(path, "w") do io
        println(io, "V_rhe,x_um,pH,cH,cOH,cHCO3,cCO3,cCO2,cCs,R_CO2R,R_HER")
        for (V, u, ps, pl) in samples
            c = unpack_log_conc(u, m.N)
            for ix in 1:m.N
                cix = c[:, ix]
                pH  = -log10(cix[1] / 1000.0)
                cCs = cCs_from_neutrality(cix)
                r1  = R_CO2R(cix[5], ps[ix], pl[ix], cix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
                r2  = R_HER(ps[ix], pl[ix], cix[1], a_v_CL, n_e)
                @printf(io, "%.6f,%.5f,%.5f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                        V, m.x_centers[ix]*1e6, pH,
                        cix[1], cix[2], cix[3], cix[4], cix[5], cCs, r1, r2)
            end
        end
    end
end

# Gas-phase profiles across GDL+CL. `samples` = Vector of (V_rhe, u).
function write_gas_profiles(path::AbstractString, gm::GasMesh, N_CL::Int, samples)
    open(path, "w") do io
        println(io, "V_rhe,x_um,pCO2,pCO,pH2,ptot")
        for (V, u) in samples
            for g in 1:gm.Ng
                ω = gas_omega(u, N_CL, g); x = omega_to_x(ω)
                p = gas_p(u, N_CL, g)
                @printf(io, "%.6f,%.5f,%.6e,%.6e,%.6e,%.6e\n",
                        V, gm.x_centers[g]*1e6, x[1]*p/101325, x[2]*p/101325,
                        x[3]*p/101325, p/101325)
            end
        end
    end
end

# Stage-5 potential profiles: φ_s across GDL+CL, φ_L across CL. `samples`=(V_rhe,u).
function write_potentials(path::AbstractString, gm::GasMesh, ly, samples)
    open(path, "w") do io
        println(io, "V_rhe,x_um,phi_s,phi_L")
        for (V, u) in samples
            for s in 1:gm.Ng
                phis = u[phi_s_idx5(ly, s)]
                phiL = s > gm.N_GDL ? u[phi_L_idx5(ly, s - gm.N_GDL)] : NaN
                @printf(io, "%.6f,%.5f,%.6e,%s\n", V, gm.x_centers[s]*1e6, phis,
                        isnan(phiL) ? "nan" : @sprintf("%.6e", phiL))
            end
        end
    end
end

# Pick ~nsel evenly-spaced voltages from a sorted (descending |V|) list of V_rhe.
function select_profile_voltages(Vs::Vector{Float64}; nsel::Int = 9)
    n = length(Vs)
    n <= nsel && return collect(1:n)
    return unique(round.(Int, range(1, n, length = nsel)))
end

nothing
