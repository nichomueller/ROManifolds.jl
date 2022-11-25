include("config_rb.jl")

function run_RB(ϵₛ::Float, ϵₜ::Float)

  RBInfo, RBVars = config_RB(ϵₛ, ϵₜ)

  offline_phase(RBInfo, RBVars)

  param_nbs = [95,96]
  online_phase(RBInfo, RBVars, param_nbs)

end

ϵₛ, ϵₜ = 1e-5, 1e-5
run_RB(ϵₛ, ϵₜ)

post_process(Paths.current_test)
