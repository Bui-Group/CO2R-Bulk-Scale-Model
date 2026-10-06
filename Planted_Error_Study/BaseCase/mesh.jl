# =============================================================================
# mesh.jl — cell-centered finite-volume meshes (BaseCase.md §9.1)
# Graded geometric mesh; CL refined toward x=L_CL (electrolyte side),
# GDL refined toward x=0 (the GDL/CL interface). Both refine the right edge.
# =============================================================================

struct Mesh
    N::Int                      # number of cells
    L::Float64                  # domain length
    x_faces::Vector{Float64}    # length N+1, face positions
    x_centers::Vector{Float64}  # length N, cell centers
    dx::Vector{Float64}         # length N, cell widths
    dx_between::Vector{Float64} # length N+1: [1]=left half (face->c1), [f]=c_{f-1}->c_f, [N+1]=right half
end

"""
    make_graded_mesh(N, x_left, x_right; q=1.3, refine=:right)

Geometric-progression cell widths. With `refine=:right` the smallest cells are
at the right edge (high index) and the ratio of largest/smallest cell width is `q`.
"""
function make_graded_mesh(N::Int, x_left::Float64, x_right::Float64; q::Float64=1.3, refine::Symbol=:right)
    @assert N >= 1
    L = x_right - x_left
    r = q^(1/(N-1))                         # per-cell width ratio (§9.1)
    # base widths: g[i] = r^(N-i)  -> g[1]=q (largest, left), g[N]=1 (smallest, right)
    g = [r^(N-i) for i in 1:N]
    if refine == :left
        reverse!(g)                          # smallest at left instead
    end
    G = sum(g)
    widths = g .* (L / G)

    x_faces = Vector{Float64}(undef, N+1)
    x_faces[1] = x_left
    for k in 1:N
        x_faces[k+1] = x_faces[k] + widths[k]
    end
    x_faces[N+1] = x_right                    # pin exactly (avoid roundoff drift)

    x_centers = [0.5*(x_faces[i] + x_faces[i+1]) for i in 1:N]
    dx = [x_faces[i+1] - x_faces[i] for i in 1:N]

    dx_between = Vector{Float64}(undef, N+1)
    dx_between[1] = x_centers[1] - x_faces[1]                  # left half-cell
    for f in 2:N
        dx_between[f] = x_centers[f] - x_centers[f-1]          # center-to-center
    end
    dx_between[N+1] = x_faces[N+1] - x_centers[N]              # right half-cell

    return Mesh(N, L, x_faces, x_centers, dx, dx_between)
end

# CL occupies x ∈ [0, L_CL], refined toward x=L_CL (electrolyte side).
make_cl_mesh(N=N_CL; q=MESH_Q) = make_graded_mesh(N, 0.0, L_CL; q=q, refine=:right)

# GDL occupies x ∈ [-L_GDL, 0], refined toward x=0 (the GDL/CL interface).
make_gdl_mesh(N=N_GDL; q=MESH_Q) = make_graded_mesh(N, -L_GDL, 0.0; q=q, refine=:right)
