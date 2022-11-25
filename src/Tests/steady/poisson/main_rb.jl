include("config_rb.jl")

function run_RB(ϵₛ::Float)

  RBInfo, RBVars = config_RB(ϵₛ)

  offline_phase(RBInfo, RBVars)

  param_nbs = [95,96]
  online_phase(RBInfo, RBVars, param_nbs)

end

ϵₛ = 1e-5
run_RB(ϵₛ)

post_process(Paths.current_test)
