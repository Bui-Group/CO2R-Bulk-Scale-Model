# mesh.jl — Cell-centered finite-volume graded mesh (§9.1)
# CL: x ∈ [0, L_CL], refined toward x = L_CL (electrolyte side).
# GDL: x ∈ [-L_GDL, 0], refined toward x = 0 (CL interface).

struct Mesh
    N::Int
    L::Float64
    x_faces::Vector{Float64}     # length N+1
    x_centers::Vector{Float64}   # length N
    dx::Vector{Float64}          # length N (cell widths)
    dx_cc::Vector{Float64}       # length N-1 (interior center-to-center)
    dx_face_left::Float64        # face(1) → center(1)
    dx_face_right::Float64       # center(N) → face(N+1)
    x0::Float64                  # left coordinate of the domain
end

"""
    make_graded_mesh(N, L; q=1.3, refine_right=true, x0=0.0)

Geometric grading. `q` is the ratio of the largest cell to the smallest cell
across the mesh (§9.1: `r = q^(1/(N-1))` per-cell ratio). `refine_right=true`
puts the smallest cells at the right end of the domain.
"""
function make_graded_mesh(N::Int, L::Float64; q::Float64=1.3,
                          refine_right::Bool=true, x0::Float64=0.0)
    if N == 1
        widths = [L]
    else
        # per-step ratio; refine_right ⇒ widths shrink toward k=N
        s = refine_right ? q^(-1/(N-1)) : q^(1/(N-1))
        powers = [s^(k-1) for k in 1:N]
        base = L / sum(powers)
        widths = base .* powers
    end
    x_faces = zeros(N+1); x_faces[1] = x0
    @inbounds for k in 1:N
        x_faces[k+1] = x_faces[k] + widths[k]
    end
    x_centers = [0.5*(x_faces[k] + x_faces[k+1]) for k in 1:N]
    dx = widths
    dx_cc = [x_centers[k+1] - x_centers[k] for k in 1:N-1]
    dx_face_left  = x_centers[1] - x_faces[1]
    dx_face_right = x_faces[N+1] - x_centers[N]
    return Mesh(N, L, x_faces, x_centers, dx, dx_cc, dx_face_left, dx_face_right, x0)
end

make_cl_mesh(N_CL::Int; q::Float64=1.3) =
    make_graded_mesh(N_CL, L_CL; q=q, refine_right=true, x0=0.0)

make_gdl_mesh(N_GDL::Int; q::Float64=1.3) =
    make_graded_mesh(N_GDL, L_GDL; q=q, refine_right=true, x0=-L_GDL)
