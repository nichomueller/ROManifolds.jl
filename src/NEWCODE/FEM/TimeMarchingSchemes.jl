abstract type TimeInfo end

struct ThetaMethodInfo <: TimeInfo
  t0::Float
  tF::Float
  dt::Float
  θ::Float

  function ThetaMethodInfo(t0::Real,tF::Real,dt::Real,θ::Real)
    new(Float(t0),Float(tF),Float(dt),Float(θ))
  end
end

get_dt(ti::TimeInfo) = ti.dt

get_Nt(ti::TimeInfo) = Int((ti.tF-ti.t0)/ti.dt)

get_θ(ti::ThetaMethodInfo) = ti.θ

get_times(ti::ThetaMethodInfo) = collect(ti.t0:ti.dt:ti.tF-ti.dt).+ti.dt*ti.θ

realization(ti::TimeInfo) = rand(Uniform(ti.t0,ti.tF))

function ThetaMethod(nls::NonlinearSolver,ti::ThetaMethodInfo)
  ThetaMethod(nls,get_dt(ti),get_θ(ti))
end
