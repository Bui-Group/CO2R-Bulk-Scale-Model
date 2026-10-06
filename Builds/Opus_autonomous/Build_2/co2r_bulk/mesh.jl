# mesh.jl — cell-centered FV meshes (§9.1)

struct Mesh1D
    N::Int
    L::Float64
    x_faces::Vector{Float64}    # N+1
    x_centers::Vector{Float64}  # N
    dx::Vector{Float64}         # N   cell widths
    dxc::Vector{Float64}        # N+1 face-to-adjacent-center distances:
                                #   dxc[1]   = x_c[1] − x_f[1]
                                #   dxc[f]   = x_c[f] − x_c[f-1]   (2 ≤ f ≤ N)
                                #   dxc[N+1] = x_f[N+1] − x_c[N]
end

"""
    graded_mesh(N, x0, x1; q=1.3, fine=:right)

Geometric-progression mesh.  `q` is the RATIO OF LARGEST TO SMALLEST cell width
(so q = 1.3 is a mild 30 % spread), per §9.1.  `fine = :right` puts the smallest
cell at `x1`.
"""
function graded_mesh(N::Int, x0::Float64, x1::Float64; q::Float64=1.3, fine::Symbol=:right)
    @assert N ≥ 2
    r = q^(1 / (N - 1))
    w = fine === :right ? [r^(N - i) for i in 1:N] : [r^(i - 1) for i in 1:N]
    L = x1 - x0
    dx = L .* w ./ sum(w)
    xf = Vector{Float64}(undef, N + 1)
    xf[1] = x0
    for i in 1:N
        xf[i+1] = xf[i] + dx[i]
    end
    xf[end] = x1                                  # kill roundoff
    xc = [0.5 * (xf[i] + xf[i+1]) for i in 1:N]
    dxc = Vector{Float64}(undef, N + 1)
    dxc[1] = xc[1] - xf[1]
    for f in 2:N
        dxc[f] = xc[f] - xc[f-1]
    end
    dxc[N+1] = xf[N+1] - xc[N]
    return Mesh1D(N, L, xf, xc, dx, dxc)
end

# CL: x ∈ [0, L_CL], refined toward the electrolyte face x = L_CL
make_cl_mesh(N_CL::Int; q::Float64=MESH_Q) = graded_mesh(N_CL, 0.0, L_CL; q=q, fine=:right)

# GDL: x ∈ [−L_GDL, 0], refined toward the GDL/CL interface x = 0
make_gdl_mesh(N_GDL::Int; q::Float64=MESH_Q) = graded_mesh(N_GDL, -L_GDL, 0.0; q=q, fine=:right)
