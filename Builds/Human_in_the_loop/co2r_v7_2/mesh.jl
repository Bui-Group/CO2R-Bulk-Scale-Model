# =============================================================================
# mesh.jl — cell-centred FV meshes for the CL and the GDL (§9.1)
# =============================================================================

"""
    FVMesh

Cell-centred 1D finite-volume mesh.

* `x_faces`  — N+1 face positions [m]
* `x_cent`   — N cell-centre positions [m]
* `dx`       — N cell widths [m]
* `dx_between` — **N+1** entries, one per face: the distance the flux at that
  face is taken across.  `dx_between[1] = dx[1]/2` (left boundary face to
  centre 1), `dx_between[f] = x_cent[f] - x_cent[f-1]` for interior faces
  `2..N`, and `dx_between[N+1] = dx[N]/2` (centre N to right boundary face).

  The face-indexed convention is what §11 Stage 5 requires for the φ_s
  ghost-cell BC (`dx_half = gdl_mesh.dx_between[1]`, "the face-to-centre
  distance"), and it is what the interior-face diffusion stencil needs at
  face `f`.
"""
struct FVMesh
    N::Int
    L::Float64
    x_faces::Vector{Float64}
    x_cent::Vector{Float64}
    dx::Vector{Float64}
    dx_between::Vector{Float64}
    x0::Float64                 # physical coordinate of face 1
end

"""
    graded_mesh(N, L; q = 1.3, x0 = 0.0)

Geometric mesh whose cells shrink monotonically with increasing index, so the
mesh is refined at the **high-index end** of the domain.  `q` is the ratio of
largest to smallest cell width (§9.1): with `r = q^(1/(N-1))` the coarsest
cell is exactly `q` times the finest.

* CL  (x0 = 0):       refined toward `x = L_CL`  (the electrolyte face)
* GDL (x0 = -L_GDL):  refined toward `x = 0`     (the GDL/CL interface)
"""
function graded_mesh(N::Int, L::Float64; q::Float64 = 1.3, x0::Float64 = 0.0)
    @assert N >= 2
    r = q^(1 / (N - 1))
    w = [r^(N - i) for i in 1:N]           # w[1] = q (coarsest) ... w[N] = 1
    dx = L .* w ./ sum(w)
    x_faces = Vector{Float64}(undef, N + 1)
    x_faces[1] = x0
    for i in 1:N
        x_faces[i+1] = x_faces[i] + dx[i]
    end
    x_faces[end] = x0 + L                  # kill round-off drift
    x_cent = [0.5 * (x_faces[i] + x_faces[i+1]) for i in 1:N]
    dxb = Vector{Float64}(undef, N + 1)
    dxb[1] = x_cent[1] - x_faces[1]
    for f in 2:N
        dxb[f] = x_cent[f] - x_cent[f-1]
    end
    dxb[N+1] = x_faces[N+1] - x_cent[N]
    return FVMesh(N, L, x_faces, x_cent, dx, dxb, x0)
end

make_cl_mesh(N_cl::Int = N_CL, L::Float64 = L_CL; q::Float64 = MESH_Q) =
    graded_mesh(N_cl, L; q = q, x0 = 0.0)

make_gdl_mesh(N_gdl::Int = N_GDL, L::Float64 = L_GDL; q::Float64 = MESH_Q) =
    graded_mesh(N_gdl, L; q = q, x0 = -L)

# ── Combined gas grid: GDL cells 1..N_GDL then CL cells 1..N_CL ──────────────
"""
    GasGrid

The gas phase spans GDL + CL as one continuous domain (§6).  Cells
`1..N_GDL` are the GDL, cells `N_GDL+1..N_GDL+N_CL` are the CL.  Face `f`
separates cell `f-1` from cell `f`; face `N_GDL+1` is the GDL/CL interface,
where the Bruggeman-corrected binary diffusivities and the Darcy permeability
are harmonic-mean averaged (§8.7, §11 Stage 4).
"""
struct GasGrid
    N::Int                        # N_GDL + N_CL
    N_GDL::Int
    N_CL::Int
    dx::Vector{Float64}           # N
    dxc::Vector{Float64}          # N+1, distance across each face
    x_cent::Vector{Float64}       # N
    Dface::Vector{Matrix{Float64}}   # N+1 binary-diffusivity matrices
    kappa::Vector{Float64}        # N+1 face permeabilities
end

function make_gas_grid(gdl::FVMesh, cl::FVMesh)
    Ng, Nc = gdl.N, cl.N
    N = Ng + Nc
    dx = vcat(gdl.dx, cl.dx)
    x_cent = vcat(gdl.x_cent, cl.x_cent)
    dxc = Vector{Float64}(undef, N + 1)
    dxc[1] = gdl.dx_between[1]
    for f in 2:Ng
        dxc[f] = gdl.dx_between[f]
    end
    dxc[Ng+1] = gdl.dx_between[Ng+1] + cl.dx_between[1]   # across the interface
    for f in 2:Nc
        dxc[Ng+f] = cl.dx_between[f]
    end
    dxc[N+1] = cl.dx_between[Nc+1]

    Dh = 2.0 .* D_bin_GDL .* D_bin_CL ./ (D_bin_GDL .+ D_bin_CL .+ 1e-300)
    kap_h = 2.0 * kappa_GDL * kappa_CL / (kappa_GDL + kappa_CL)

    Dface = Vector{Matrix{Float64}}(undef, N + 1)
    kappa = Vector{Float64}(undef, N + 1)
    for f in 1:(N+1)
        if f <= Ng                    # faces inside / at the left of the GDL
            Dface[f] = D_bin_GDL; kappa[f] = kappa_GDL
        elseif f == Ng + 1            # GDL/CL interface
            Dface[f] = Dh;         kappa[f] = kap_h
        else                          # faces inside / at the right of the CL
            Dface[f] = D_bin_CL;   kappa[f] = kappa_CL
        end
    end
    return GasGrid(N, Ng, Nc, dx, dxc, x_cent, Dface, kappa)
end
