# mesh.jl — CLMesh, GDLMesh: graded 1D finite-volume meshes (§9.1, §8.2).
#
# dx_between convention (a deliberate unification of two inconsistent usages in the guide —
# see BUILD_DIALOGUE.md item 1): length N+1, indexed by FACE number f = 1..N+1.
#   dx_between[1]     = distance from the left boundary face to cell-1 center
#   dx_between[2..N]  = center-to-center distance between cell f-1 and cell f
#   dx_between[N+1]   = distance from cell-N center to the right boundary face

struct FVMesh
    N::Int
    x_faces::Vector{Float64}      # N+1
    x_centers::Vector{Float64}    # N
    dx::Vector{Float64}           # N, cell widths
    dx_between::Vector{Float64}   # N+1, face-associated distances (see above)
end

"""
    make_graded_mesh(N, L, q; x0=0.0)

Geometric mesh on [x0, x0+L] with N cells, cell 1 the LARGEST and cell N the SMALLEST
(ratio of largest to smallest width = q). This is the CL grading (refined toward x=L_CL,
i.e. toward cell N) used directly; the GDL mesh reuses the same construction with
x0 = -L_GDL, which places its smallest cell at the GDL/CL interface (x=0), matching the
guide's "GDL refined toward x=0" requirement.
"""
function make_graded_mesh(N::Int, L::Float64, q::Float64; x0::Float64=0.0)
    r = q^(1.0 / (N - 1))
    what = [r^(-(i - 1)) for i in 1:N]          # unnormalized widths, decreasing
    dx = L .* what ./ sum(what)

    x_faces = zeros(N + 1)
    x_faces[1] = x0
    for i in 1:N
        x_faces[i+1] = x_faces[i] + dx[i]
    end
    x_centers = [0.5 * (x_faces[i] + x_faces[i+1]) for i in 1:N]

    dx_between = zeros(N + 1)
    dx_between[1] = x_centers[1] - x_faces[1]
    for i in 2:N
        dx_between[i] = x_centers[i] - x_centers[i-1]
    end
    dx_between[N+1] = x_faces[N+1] - x_centers[N]

    return FVMesh(N, x_faces, x_centers, dx, dx_between)
end

make_cl_mesh(N_CL::Int; q::Float64=MESH_Q)   = make_graded_mesh(N_CL, L_CL, q; x0=0.0)
make_gdl_mesh(N_GDL::Int; q::Float64=MESH_Q) = make_graded_mesh(N_GDL, L_GDL, q; x0=-L_GDL)

"Center-to-center distance across the GDL/CL interface (x=0), spanning both meshes."
gdl_cl_interface_dx(gdl_mesh::FVMesh, cl_mesh::FVMesh) =
    gdl_mesh.dx_between[gdl_mesh.N + 1] + cl_mesh.dx_between[1]
