##############################################################################
# mesh.jl — graded 1D finite-volume meshes (§9.1)
##############################################################################

struct Mesh1D
    N::Int
    L::Float64
    x_face::Vector{Float64}      # length N+1, local coords [0, L]
    x_center::Vector{Float64}    # length N
    dx::Vector{Float64}          # length N, cell widths
    dx_between::Vector{Float64}  # length N-1, interior center-to-center distances
    dx_left::Float64             # boundary face(1) -> center(1)
    dx_right::Float64            # center(N) -> boundary face(N+1)
end

"""
    graded_mesh(N, L; q=1.3, refine_right=true)

Geometric-progression mesh on [0, L]. `q` is the ratio of the LARGEST to the
SMALLEST cell width (§9.1: "q is the ratio of largest to smallest cell width,
not a per-cell ratio"). `refine_right=true` puts the smallest cells at the
local right end (x = L); `refine_right=false` puts them at the local left end
(x = 0).
"""
function graded_mesh(N::Int, L::Float64; q::Float64=1.3, refine_right::Bool=true)
    if N == 1
        w = [L]
    else
        r = q^(1/(N-1))           # per-cell ratio; r^(N-1) = q
        # raw[ix] = r^(N-ix): raw[1] = r^(N-1) = q (largest), raw[N] = r^0 = 1 (smallest)
        raw = [r^(N - ix) for ix in 1:N]
        if !refine_right
            raw = reverse(raw)
        end
        w = raw ./ sum(raw) .* L
    end
    x_face = zeros(N+1)
    for ix in 1:N
        x_face[ix+1] = x_face[ix] + w[ix]
    end
    x_center = [0.5*(x_face[ix] + x_face[ix+1]) for ix in 1:N]
    dx_between = [x_center[ix+1] - x_center[ix] for ix in 1:N-1]
    dx_left  = x_center[1] - x_face[1]
    dx_right = x_face[N+1] - x_center[N]
    return Mesh1D(N, L, x_face, x_center, w, dx_between, dx_left, dx_right)
end

# CL: refined toward x = L_CL (electrolyte side) — smallest cells at ix = N_CL.
make_cl_mesh(N::Int=N_CL; q::Float64=MESH_Q) = graded_mesh(N, L_CL; q=q, refine_right=true)

# GDL: local coords run collector (ix=1, x_global=-L_GDL) -> interface (ix=N_GDL, x_global=0).
# "Refined toward x=0" means smallest cells at ix = N_GDL, i.e. refine_right=true in local coords.
make_gdl_mesh(N::Int=N_GDL; q::Float64=MESH_Q) = graded_mesh(N, L_GDL; q=q, refine_right=true)

# Global x coordinate for a GDL mesh (x in [-L_GDL, 0])
gdl_x_global(m::Mesh1D) = m.x_center .- m.L
gdl_xface_global(m::Mesh1D) = m.x_face .- m.L

# CL global x coordinate is just x_center (domain already [0, L_CL])
cl_x_global(m::Mesh1D) = m.x_center
