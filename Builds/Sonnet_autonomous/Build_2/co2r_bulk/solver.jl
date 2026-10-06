# solver.jl — legacy dense FD Jacobian (§9.3.1). Small-mesh bring-up / debugging ONLY; not used
# by any production driver (all of which use the ForwardDiff AD Jacobian via newton_affine.jl).

function fd_jacobian!(J::Matrix{Float64}, res!::Function, F0::Vector{Float64}, u::Vector{Float64};
                       eps::Float64=1e-7)
    N = length(u)
    u_p = copy(u)
    F_p = similar(F0)
    for j in 1:N
        h = eps * max(abs(u[j]), 1.0)
        u_p[j] = u[j] + h
        res!(F_p, u_p)
        @. J[:, j] = (F_p - F0) / h
        u_p[j] = u[j]
    end
    return J
end
