# mesh.jl — cell-centered finite-volume meshes for CL and GDL (§9.1)
# Graded geometric mesh. CL refined toward x = L_CL (electrolyte side);
# GDL refined toward x = 0 (GDL/CL interface).

struct FVMesh
    N::Int
    xf::Vector{Float64}          # faces,        length N+1
    xc::Vector{Float64}          # cell centers, length N
    dx::Vector{Float64}          # cell widths,  length N   (= diff(xf))
    dx_between::Vector{Float64}  # center-to-center distance, length N-1 (k connects cell k,k+1)
    dx_face_left::Float64        # face1 → center1
    dx_face_right::Float64       # centerN → faceN+1
end

"""
    make_mesh(N, x_lo, x_hi; q=1.3, refine=:right)

Geometric-graded FV mesh on [x_lo, x_hi]. `refine=:right` makes cells smallest at
x_hi (largest at x_lo); `refine=:left` is the mirror image. `q` is the total
size ratio (largest/smallest cell ≈ q), per guide §9.1 (q ≈ 1.3 production).
"""
function make_mesh(N::Int, x_lo::Float64, x_hi::Float64; q::Float64=1.3, refine::Symbol=:right)
    L = x_hi - x_lo
    r = q^(1 / (N - 1))                       # per-step growth ratio (§9.1)
    # widths largest at i=1, smallest at i=N : w_i ∝ r^(N-i)
    w = [r^(N - i) for i in 1:N]
    w .*= L / sum(w)                          # normalize to total length
    if refine == :left
        reverse!(w)                           # smallest at i=1 instead
    end
    xf = Vector{Float64}(undef, N + 1)
    xf[1] = x_lo
    for i in 1:N
        xf[i+1] = xf[i] + w[i]
    end
    xf[end] = x_hi                            # guard against float drift
    xc = [0.5 * (xf[i] + xf[i+1]) for i in 1:N]
    dx = diff(xf)
    dx_between = [xc[i+1] - xc[i] for i in 1:N-1]
    dx_face_left  = xc[1] - xf[1]
    dx_face_right = xf[N+1] - xc[N]
    return FVMesh(N, xf, xc, dx, dx_between, dx_face_left, dx_face_right)
end

# CL occupies x ∈ [0, L_CL], refined toward L_CL (electrolyte).
make_cl_mesh(N::Int=N_CL; q::Float64=1.3) = make_mesh(N, 0.0, L_CL; q=q, refine=:right)

# GDL occupies x ∈ [-L_GDL, 0], refined toward 0 (GDL/CL interface = right end).
make_gdl_mesh(N::Int=N_GDL; q::Float64=1.3) = make_mesh(N, -L_GDL, 0.0; q=q, refine=:right)
