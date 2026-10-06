# =====================================================================
#  mesh.jl — cell-centred FV meshes (§9.1)
# =====================================================================

"""
    Mesh1D

Cell-centred 1-D finite-volume mesh.

* `x_faces`   – N+1 face coordinates [m]
* `x_centers` – N cell-centre coordinates [m]
* `dx`        – N cell widths [m]
* `dx_between`– N+1 face-normal distances used by the face fluxes:
                `dx_between[1]`   = left boundary face → centre of cell 1
                `dx_between[f]`   = centre(f-1) → centre(f)   for f = 2..N
                `dx_between[N+1]` = centre of cell N → right boundary face
"""
struct Mesh1D
    N::Int
    x_faces::Vector{Float64}
    x_centers::Vector{Float64}
    dx::Vector{Float64}
    dx_between::Vector{Float64}
end

"""
    graded_widths(N, L; q = 1.3, refine_right = true)

Geometric progression of `N` cell widths summing to `L`.  `q` is the ratio of
the largest to the smallest cell (`q = 1` ⇒ uniform).  With
`refine_right = true` the cells shrink toward the right end of the interval.
"""
function graded_widths(N::Int, L::Float64; q::Float64 = 1.3, refine_right::Bool = true)
    N == 1 && return [L]
    r = q^(1 / (N - 1))                       # per-cell ratio, r^(N-1) = q
    w = [r^(-(i - 1)) for i in 1:N]           # w[1] largest, w[N] = w[1]/q
    refine_right || (w = reverse(w))
    w .*= L / sum(w)
    return w
end

function build_mesh(N::Int, x_left::Float64, x_right::Float64;
                    q::Float64 = 1.3, refine_right::Bool = true)
    L  = x_right - x_left
    dx = graded_widths(N, L; q = q, refine_right = refine_right)
    xf = Vector{Float64}(undef, N + 1)
    xf[1] = x_left
    for i in 1:N
        xf[i + 1] = xf[i] + dx[i]
    end
    xf[N + 1] = x_right                        # kill accumulated round-off
    xc = [0.5 * (xf[i] + xf[i + 1]) for i in 1:N]
    dcc = Vector{Float64}(undef, N + 1)
    dcc[1] = xc[1] - xf[1]
    for f in 2:N
        dcc[f] = xc[f] - xc[f - 1]
    end
    dcc[N + 1] = xf[N + 1] - xc[N]
    return Mesh1D(N, xf, xc, dx, dcc)
end

"CL mesh on x ∈ [0, L_CL], refined toward the electrolyte face x = L_CL (§9.1)."
make_cl_mesh(N_CL::Int = N_CL_DEFAULT; q::Float64 = MESH_Q) =
    build_mesh(N_CL, 0.0, L_CL; q = q, refine_right = true)

"GDL mesh on x ∈ [−L_GDL, 0], refined toward the GDL/CL interface x = 0 (§9.1)."
make_gdl_mesh(N_GDL::Int = N_GDL_DEFAULT; q::Float64 = MESH_Q) =
    build_mesh(N_GDL, -L_GDL, 0.0; q = q, refine_right = true)
