# ============================================================================
# mesh.jl — Cell-centered finite-volume meshes (graded), §9.1
#
# Conventions (match the residual pseudocode in §11):
#   dx[k]          : width of cell k                       (length N)
#   dx_between[k]  : center-to-center distance, cell k↔k+1 (length N-1, interior)
#   dx_face_left   : boundary-face → first-cell-center distance (half cell)
#   dx_face_right  : last-cell-center → boundary-face distance (half cell)
#   x_faces        : face coordinates                       (length N+1)
#   x_centers      : cell-center coordinates                (length N)
# ============================================================================

struct Mesh
    N::Int
    x_faces::Vector{Float64}
    x_centers::Vector{Float64}
    dx::Vector{Float64}
    dx_between::Vector{Float64}     # interior center-to-center (N-1)
    dx_face_left::Float64
    dx_face_right::Float64
end

# Geometric cell widths over a length L.
#   refine_right = true  → smallest cells at the high-index (right) end,
#                          largest at the low-index (left) end; max/min ratio = q.
#   refine_right = false → smallest cells at the left end.
function geom_widths(N::Int, L::Float64, q::Float64; refine_right::Bool)
    if N == 1
        return [L]
    end
    # ratio between adjacent widths (w_{k+1}/w_k)
    ρ = refine_right ? (1/q)^(1/(N-1)) : q^(1/(N-1))
    # w_1 * Σ_{k=0}^{N-1} ρ^k = L
    Σ = (ρ ≈ 1.0) ? float(N) : (1 - ρ^N) / (1 - ρ)
    w1 = L / Σ
    return [w1 * ρ^(k-1) for k in 1:N]
end

function build_mesh(N::Int, x_start::Float64, L::Float64, q::Float64; refine_right::Bool)
    w = geom_widths(N, L, q; refine_right = refine_right)
    x_faces = Vector{Float64}(undef, N+1)
    x_faces[1] = x_start
    for k in 1:N
        x_faces[k+1] = x_faces[k] + w[k]
    end
    x_centers = [0.5 * (x_faces[k] + x_faces[k+1]) for k in 1:N]
    dx = copy(w)
    dx_between = [x_centers[k+1] - x_centers[k] for k in 1:N-1]
    dx_face_left  = x_centers[1] - x_faces[1]
    dx_face_right = x_faces[N+1] - x_centers[N]
    return Mesh(N, x_faces, x_centers, dx, dx_between, dx_face_left, dx_face_right)
end

# CL: domain [0, L_CL], refined toward x = L_CL (electrolyte side). §9.1
make_cl_mesh(N::Int, Lcl::Float64; q::Float64 = 1.3) =
    build_mesh(N, 0.0, Lcl, q; refine_right = true)

# GDL: domain [-L_GDL, 0], refined toward x = 0 (CL interface), coarsening away. §9.1
make_gdl_mesh(N::Int, Lgdl::Float64; q::Float64 = 1.3) =
    build_mesh(N, -Lgdl, Lgdl, q; refine_right = true)
