##############################################################################
# solid_transport.jl — solid-phase (electronic) Ohm's law on GDL+CL (§3.4)
##############################################################################

"""
    solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, sigma_gdl, sigma_cl,
                         gdl_mesh, cl_mesh)

Returns i_s, a vector of length N_GDL+N_CL+1 face currents [A/m^2].
Face 1 = ghost-cell Dirichlet BC at the current collector (§3.4).
Face N_GDL+1 = GDL/CL interface (harmonic-mean sigma).
Face N_GDL+N_CL+1 = Neumann zero at the electrolyte-side CL boundary.
"""
function solid_face_currents(phi_s_gdl::AbstractVector, phi_s_cl::AbstractVector,
                              V_applied_SHE, sigma_gdl::Float64, sigma_cl::Float64,
                              gdl_mesh::Mesh1D, cl_mesh::Mesh1D)
    N_GDL = gdl_mesh.N
    N_CL  = cl_mesh.N
    i_s = similar(phi_s_gdl, N_GDL + N_CL + 1)

    # Left ghost-cell Dirichlet BC (NOT a residual override — §3.4, §11 gotcha #6)
    i_s[1] = -sigma_gdl * (phi_s_gdl[1] - V_applied_SHE) / gdl_mesh.dx_left

    # Interior GDL faces
    for ig in 1:(N_GDL-1)
        i_s[ig+1] = -sigma_gdl * (phi_s_gdl[ig+1] - phi_s_gdl[ig]) / gdl_mesh.dx_between[ig]
    end

    # GDL/CL interface (harmonic mean conductivity, §3.4)
    sigma_int = 2*sigma_gdl*sigma_cl / (sigma_gdl + sigma_cl)
    dx_int = gdl_mesh.dx_right + cl_mesh.dx_left
    i_s[N_GDL+1] = -sigma_int * (phi_s_cl[1] - phi_s_gdl[N_GDL]) / dx_int

    # Interior CL faces
    for ix in 1:(N_CL-1)
        i_s[N_GDL+1+ix] = -sigma_cl * (phi_s_cl[ix+1] - phi_s_cl[ix]) / cl_mesh.dx_between[ix]
    end

    # Right Neumann zero (electrons cannot leave the solid into the liquid except via reaction)
    i_s[N_GDL+N_CL+1] = zero(eltype(i_s))

    return i_s
end
