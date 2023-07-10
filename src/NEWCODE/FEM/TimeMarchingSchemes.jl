struct θMethod <: ODESolver
  nls::NonlinearSolver
  t0::Real
  tF::Real
  dt::Real
  θ::Real
  uh0::Function
end

get_time_ndofs(ti::ODESolver) = Int((ti.tF-ti.t0)/ti.dt)

get_times(ti::ODESolver) = collect(ti.dt:ti.dt:ti.tF)

realization(ti::ODESolver) = rand(Uniform(ti.t0,ti.tF))
