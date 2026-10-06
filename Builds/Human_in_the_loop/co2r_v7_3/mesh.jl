# ============================================================================
# mesh.jl — cell-centered FV meshes (§9.1)
# ============================================================================

struct Mesh1D
    N::Int
    L::Float64
    x_faces::Vector{Float64}      # length N+1
    x_centers::Vector{Float64}    # length N
    dx::Vector{Float64}           # length N   (cell widths)
    dx_between::Vector{Float64}   # length N+1 (FACE-indexed distance across face f)
end

"""
    make_graded_mesh(N, L; q = 1.3, x0 = 0.0)

Geometric progression refined at the RIGHT end (x = x0 + L).  `q` is the ratio
of the largest to the smallest cell width (§9.1): `r^(N-1) = q`, so q = 1.3 is
a mild 30 % spread, not a per-cell ratio.

`dx_between` is **face-indexed** (length N+1):
  dx_between[1]   = x_centers[1] - x_faces[1]        (face-to-centre, left BC)
  dx_between[f]   = x_centers[f] - x_centers[f-1]    (interior face f = 2..N)
  dx_between[N+1] = x_faces[N+1] - x_centers[N]      (face-to-centre, right BC)

Guide note: §11 Stage 1 pseudocode writes `dx_between[f-1]` for interior face
`f`, which is inconsistent with §11 Stage 5's statement that
`gdl_mesh.dx_between[1]` is the collector-face-to-first-cell-centre distance.
The face-indexed convention above satisfies the Stage-5 statement and is the
geometrically correct one; interior face `f` therefore uses `dx_between[f]`.
"""
function make_graded_mesh(N::Int, L::Float64; q::Float64 = 1.3, x0::Float64 = 0.0)
    r  = q^(1 / (N - 1))
    a  = L * (r - 1) / (r^N - 1)          # smallest cell (at the right end)
    dx = [a * r^(N - i) for i in 1:N]     # dx[1] = a*q (largest) … dx[N] = a
    dx .*= L / sum(dx)                    # exact length normalisation
    xf = zeros(N + 1)
    xf[1] = x0
    for i in 1:N
        xf[i+1] = xf[i] + dx[i]
    end
    xc = [0.5 * (xf[i] + xf[i+1]) for i in 1:N]
    dxb = zeros(N + 1)
    dxb[1] = xc[1] - xf[1]
    for f in 2:N
        dxb[f] = xc[f] - xc[f-1]
    end
    dxb[N+1] = xf[N+1] - xc[N]
    return Mesh1D(N, L, xf, xc, dx, dxb)
end

# CL: x ∈ [0, L_CL], refined toward x = L_CL (electrolyte side)
make_cl_mesh(N_CL, L; q = MESH_Q) = make_graded_mesh(N_CL, L; q = q, x0 = 0.0)

# GDL: x ∈ [-L_GDL, 0], refined toward x = 0 (the GDL/CL interface)
make_gdl_mesh(N_GDL, L; q = MESH_Q) = make_graded_mesh(N_GDL, L; q = q, x0 = -L)

"""
    GasMesh(gdl, cl)

Combined GDL+CL mesh used for the gas DOFs and for φ_s.  Cells 1..N_GDL are
GDL cells, cells N_GDL+1..N_GDL+N_CL are CL cells.  Faces:
  f = 1                : flow-field / current-collector boundary
  f = 2 .. N_GDL       : GDL interior
  f = N_GDL+1          : GDL/CL interface (harmonic-mean properties)
  f = N_GDL+2 .. N_G   : CL interior
  f = N_G+1            : electrolyte-side boundary
"""
struct GasMesh
    N_GDL::Int
    N_CL::Int
    N::Int
    dx::Vector{Float64}
    dx_between::Vector{Float64}   # length N+1, face-indexed
    x_centers::Vector{Float64}
    x_faces::Vector{Float64}
end

function GasMesh(gdl::Mesh1D, cl::Mesh1D)
    NG, NC = gdl.N, cl.N
    N = NG + NC
    dx = vcat(gdl.dx, cl.dx)
    dxb = zeros(N + 1)
    dxb[1] = gdl.dx_between[1]
    for f in 2:NG
        dxb[f] = gdl.dx_between[f]
    end
    dxb[NG+1] = gdl.dx_between[NG+1] + cl.dx_between[1]   # across the interface
    for k in 2:NC
        dxb[NG+k] = cl.dx_between[k]
    end
    dxb[N+1] = cl.dx_between[NC+1]
    xc = vcat(gdl.x_centers, cl.x_centers)
    xf = vcat(gdl.x_faces[1:end-1], cl.x_faces)
    return GasMesh(NG, NC, N, dx, dxb, xc, xf)
end

is_gdl_cell(gm::GasMesh, g::Int) = g <= gm.N_GDL
