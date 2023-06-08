struct θMethod <: ODESolver
  nls::NonlinearSolver
  t0::Real
  tF::Real
  dt::Real
  θ::Real
end

get_time_ndofs(ti::ODESolver) = Int((ti.tF-ti.t0)/ti.dt)

get_times(ti::ODESolver) = collect(ti.t0:ti.dt:ti.tF-ti.dt).+ti.dt*ti.θ

realization(ti::ODESolver) = rand(Uniform(ti.t0,ti.tF))
