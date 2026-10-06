# mesh.jl — cell-centred FV meshes with geometric grading (§9.1)

struct Mesh1D
    N::Int
    x_faces::Vector{Float64}      # N+1
    x_centers::Vector{Float64}    # N
    dx::Vector{Float64}           # N   cell widths
    dxc::Vector{Float64}          # N+1 face "cell-centre distances":
                                  #   dxc[1]     = x_c[1]   - x_face[1]
                                  #   dxc[f]     = x_c[f]   - x_c[f-1]   (2..N)
                                  #   dxc[N+1]   = x_face[N+1] - x_c[N]
end

"""
    graded_mesh(N, L, x0; q = 1.3, refine = :right)

Geometric mesh on `[x0, x0+L]`.  `q` is the ratio of LARGEST to SMALLEST cell
width (§9.1) — `q = 1.3` is a mild 30 % spread, not a per-cell ratio.
`refine = :right` puts the finest cells at the high-x end.
"""
function graded_mesh(N::Int, L::Float64, x0::Float64; q::Float64 = 1.3,
                     refine::Symbol = :right)
    @assert N >= 2
    r = q^(1 / (N - 1))                       # dx[i+1]/dx[i] = 1/r when refining right
    w = [r^(-(i - 1)) for i in 1:N]           # decreasing widths (fine on the right)
    refine === :left && (w = reverse(w))
    dx = L .* w ./ sum(w)

    x_faces = zeros(N + 1)
    x_faces[1] = x0
    for i in 1:N
        x_faces[i+1] = x_faces[i] + dx[i]
    end
    x_faces[N+1] = x0 + L                     # kill round-off drift
    x_centers = 0.5 .* (x_faces[1:N] .+ x_faces[2:N+1])

    dxc = zeros(N + 1)
    dxc[1] = x_centers[1] - x_faces[1]
    for f in 2:N
        dxc[f] = x_centers[f] - x_centers[f-1]
    end
    dxc[N+1] = x_faces[N+1] - x_centers[N]

    return Mesh1D(N, x_faces, x_centers, dx, dxc)
end

"CL mesh: x ∈ [0, L_CL], refined toward x = L_CL (electrolyte side)."
make_cl_mesh(N_CL::Int = N_CL_DEFAULT; q = MESH_Q) =
    graded_mesh(N_CL, L_CL, 0.0; q = q, refine = :right)

"GDL mesh: x ∈ [−L_GDL, 0], refined toward x = 0 (the GDL/CL interface)."
make_gdl_mesh(N_GDL::Int = N_GDL_DEFAULT; q = MESH_Q) =
    graded_mesh(N_GDL, L_GDL, -L_GDL; q = q, refine = :right)

"""
    GasMesh

The gas domain is the concatenation GDL ∥ CL: cells `1:N_GDL` are the GDL,
cells `N_GDL+1 : N_GDL+N_CL` are the CL.  Faces run `1 : N+1`; face
`N_GDL+1` is the GDL/CL interface (harmonic-mean transport coefficients).
"""
struct GasMesh
    N::Int
    N_GDL::Int
    N_CL::Int
    x_centers::Vector{Float64}
    x_faces::Vector{Float64}
    dx::Vector{Float64}
    dxc::Vector{Float64}
    i_int::Int                    # face index of the GDL/CL interface
end

function make_gas_mesh(gdl::Mesh1D, cl::Mesh1D)
    N = gdl.N + cl.N
    x_faces   = vcat(gdl.x_faces, cl.x_faces[2:end])
    x_centers = vcat(gdl.x_centers, cl.x_centers)
    dx        = vcat(gdl.dx, cl.dx)
    dxc = zeros(N + 1)
    dxc[1] = x_centers[1] - x_faces[1]
    for f in 2:N
        dxc[f] = x_centers[f] - x_centers[f-1]
    end
    dxc[N+1] = x_faces[N+1] - x_centers[N]
    return GasMesh(N, gdl.N, cl.N, x_centers, x_faces, dx, dxc, gdl.N + 1)
end
