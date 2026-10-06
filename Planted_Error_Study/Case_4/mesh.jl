# mesh.jl — cell-centered graded FV meshes (§9.1)
#
# Mesh fields:
#   N          number of cells
#   x_faces    (N+1) face positions [m]
#   x_centers  (N)   cell-center positions [m]
#   dx         (N)   cell widths [m]
#   dx_cc      (N+1) distance used by the flux at each face:
#                    dx_cc[1]   = x_centers[1] - x_faces[1]   (left half-cell)
#                    dx_cc[f]   = x_centers[f] - x_centers[f-1] for interior faces 2..N
#                    dx_cc[N+1] = x_faces[N+1] - x_centers[N]  (right half-cell)

struct Mesh
    N::Int
    x_faces::Vector{Float64}
    x_centers::Vector{Float64}
    dx::Vector{Float64}
    dx_cc::Vector{Float64}
end

# Geometric mesh on [x_lo, x_hi], refined at the RIGHT end (smallest cell at x_hi),
# cells growing toward x_lo. `q` = total (largest/smallest) width ratio; per-cell
# ratio is r = q^(1/(N-1)) per §9.1 (`r = q^(1/(N-1))`).
function make_mesh(N::Int, x_lo::Float64, x_hi::Float64; q::Float64 = 1.3)
    L = x_hi - x_lo
    if N == 1
        w = [L]
    else
        r = q^(1 / (N - 1))
        # w[i] = w_min * r^(N-i): largest at i=1 (left), smallest at i=N (right)
        weights = [r^(N - i) for i in 1:N]
        w = weights .* (L / sum(weights))
    end
    x_faces = Vector{Float64}(undef, N + 1)
    x_faces[1] = x_lo
    for i in 1:N
        x_faces[i + 1] = x_faces[i] + w[i]
    end
    x_faces[N + 1] = x_hi                      # guard against float drift
    x_centers = 0.5 .* (x_faces[1:N] .+ x_faces[2:N+1])
    dx = w
    dx_cc = Vector{Float64}(undef, N + 1)
    dx_cc[1] = x_centers[1] - x_faces[1]
    for f in 2:N
        dx_cc[f] = x_centers[f] - x_centers[f - 1]
    end
    dx_cc[N + 1] = x_faces[N + 1] - x_centers[N]
    return Mesh(N, x_faces, x_centers, dx, dx_cc)
end

# CL occupies x ∈ [0, L_CL], refined toward x = L_CL (electrolyte side).
make_cl_mesh(; q = MESH_Q) = make_mesh(N_CL, 0.0, L_CL; q = q)

# GDL occupies x ∈ [−L_GDL, 0], refined toward x = 0 (GDL/CL interface).
make_gdl_mesh(; q = MESH_Q) = make_mesh(N_GDL, -L_GDL, 0.0; q = q)

nothing
