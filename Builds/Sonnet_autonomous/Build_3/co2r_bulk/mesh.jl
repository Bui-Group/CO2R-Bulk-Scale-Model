# mesh.jl — CLMesh, GDLMesh structs; graded mesh construction, §9.1

struct Mesh1D
    N::Int
    x_faces::Vector{Float64}     # N+1 face positions (physical, ascending)
    x_centers::Vector{Float64}   # N cell centers
    dx::Vector{Float64}          # N cell widths
    dx_between::Vector{Float64}  # N-1 distances between adjacent centers
    dx_face_left::Float64        # distance from left boundary face to center 1
    dx_face_right::Float64       # distance from center N to right boundary face
end

"""
    graded_widths(N, L; q=1.3)

Geometric progression of N cell widths summing to L, with dx[1] the LARGEST
and dx[N] the SMALLEST (ratio dx[1]/dx[N] = q). §9.1: "q is the ratio of
largest to smallest cell width... q > 1: cells get larger toward x=0."
"""
function graded_widths(N::Int, L::Float64; q::Float64=1.3)
    if N == 1
        return [L]
    end
    r = q^(1 / (N - 1))       # per-cell ratio; dx[i] = dx1 / r^(i-1)
    ratio = 1 / r             # < 1
    # sum_{i=0}^{N-1} dx1 * ratio^i = dx1 * (1 - ratio^N)/(1 - ratio) = L
    dx1 = L * (1 - ratio) / (1 - ratio^N)
    dx = [dx1 * ratio^(i - 1) for i in 1:N]
    # Renormalize for floating-point drift so sum is exactly L
    dx .*= L / sum(dx)
    return dx
end

"""
    make_mesh(N, L; q=1.3, x0=0.0)

Build a Mesh1D on [x0, x0+L] with cell 1 (near x0) COARSEST and cell N
(near x0+L) FINEST — matches the CL grading (§8.2, §9.1: refined toward
x = L_CL). For the GDL, build on [0, L_GDL] then shift with x0 = -L_GDL so
physical x=0 (GDL/CL interface) coincides with the finest GDL cell (§9.1:
"GDL refined toward x=0 at the GDL/CL interface").
"""
function make_mesh(N::Int, L::Float64; q::Float64=1.3, x0::Float64=0.0)
    dx = graded_widths(N, L; q=q)
    x_faces = Vector{Float64}(undef, N + 1)
    x_faces[1] = x0
    for i in 1:N
        x_faces[i+1] = x_faces[i] + dx[i]
    end
    x_faces[end] = x0 + L   # pin exactly
    x_centers = [0.5 * (x_faces[i] + x_faces[i+1]) for i in 1:N]
    dx_between = [x_centers[i+1] - x_centers[i] for i in 1:N-1]
    dx_face_left = x_centers[1] - x_faces[1]
    dx_face_right = x_faces[end] - x_centers[end]
    return Mesh1D(N, x_faces, x_centers, dx, dx_between, dx_face_left, dx_face_right)
end

make_cl_mesh(N_CL::Int=MESH_N_CL; q::Float64=MESH_Q) = make_mesh(N_CL, L_CL; q=q, x0=0.0)
make_gdl_mesh(N_GDL::Int=MESH_N_GDL; q::Float64=MESH_Q) = make_mesh(N_GDL, L_GDL; q=q, x0=-L_GDL)
