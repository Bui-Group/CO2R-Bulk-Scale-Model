# mesh.jl — cell-centered FV meshes (§9.1)
#
# Indexing convention used throughout this build (harmonised — see BUILD_DIALOGUE):
#   cells   ix = 1 … N               (dx[ix]      = cell width)
#   faces    f = 1 … N+1             (face f sits between cells f-1 and f)
#   dx_cc[f] = centre-to-centre distance across face f, with the boundary faces
#              using the half-cell distance:
#                dx_cc[1]   = x_center[1]  - x_face[1]
#                dx_cc[f]   = x_center[f]  - x_center[f-1]   (2 ≤ f ≤ N)
#                dx_cc[N+1] = x_face[N+1]  - x_center[N]

struct FVMesh
    N::Int
    x_faces::Vector{Float64}
    x_centers::Vector{Float64}
    dx::Vector{Float64}
    dx_cc::Vector{Float64}
end

"""
    graded_widths(N, L; q=1.3, fine_at_right=true)

Geometric-progression cell widths.  `q` is the ratio of the LARGEST to the
SMALLEST cell width (§9.1), so r = q^(1/(N-1)) is the per-cell ratio.
`fine_at_right=true` refines toward x = L (widths decrease with index).
"""
function graded_widths(N::Int, L::Float64; q::Float64=1.3, fine_at_right::Bool=true)
    N == 1 && return [L]
    r = q^(1/(N-1))
    # widths w_i = w_1 * r^-(i-1) when fine at right (w_1 largest)
    rel = [r^(-(i-1)) for i in 1:N]
    if !fine_at_right
        rel = reverse(rel)
    end
    return L .* rel ./ sum(rel)
end

function build_mesh(N::Int, L::Float64, x0::Float64;
                    q::Float64=1.3, fine_at_right::Bool=true)
    dx = graded_widths(N, L; q=q, fine_at_right=fine_at_right)
    xf = Vector{Float64}(undef, N+1)
    xf[1] = x0
    for i in 1:N
        xf[i+1] = xf[i] + dx[i]
    end
    xc = [0.5*(xf[i] + xf[i+1]) for i in 1:N]
    dxcc = Vector{Float64}(undef, N+1)
    dxcc[1] = xc[1] - xf[1]
    for f in 2:N
        dxcc[f] = xc[f] - xc[f-1]
    end
    dxcc[N+1] = xf[N+1] - xc[N]
    return FVMesh(N, xf, xc, dx, dxcc)
end

# CL: x ∈ [0, L_CL], refined toward x = L_CL (electrolyte side)
make_cl_mesh(N_CL::Int; q::Float64=MESH_Q) =
    build_mesh(N_CL, L_CL, 0.0; q=q, fine_at_right=true)

# GDL: x ∈ [-L_GDL, 0], refined toward x = 0 (the GDL/CL interface)
make_gdl_mesh(N_GDL::Int; q::Float64=MESH_Q) =
    build_mesh(N_GDL, L_GDL, -L_GDL; q=q, fine_at_right=true)
