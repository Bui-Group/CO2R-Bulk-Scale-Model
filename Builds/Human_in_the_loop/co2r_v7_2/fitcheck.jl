include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
include(joinpath(@__DIR__, "fitted_kinetics.jl"))
using Printf
ctx = make_ctx(4)
Vs = collect(V_RHE_grid)
CD = joinpath(@__DIR__, "output_fit", "cache")
sols = [load_solution(CD, "fit_", V) for V in Vs]
@printf("loaded %d fit-cache solutions; kinetics i0=%.4f lam=%.5f pf=%.4e ac=%.6f\n",
        count(!isnothing, sols), i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
stage3_checks(Vs, Vector{Vector{Float64}}(sols), ctx; pinned = false)
