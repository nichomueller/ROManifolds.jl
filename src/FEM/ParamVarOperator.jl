struct TimeInfo
  t0::Real
  tF::Real
  dt::Real
  θ::Real
end

get_dt(ti::TimeInfo) = ti.dt
get_Nt(ti::TimeInfo) = Int((ti.tF-ti.t0)/ti.dt)
get_θ(ti::TimeInfo) = ti.θ
get_timesθ(ti::TimeInfo) = collect(ti.t0:ti.dt:ti.tF-ti.dt).+ti.dt*ti.θ
realization(ti::TimeInfo) = rand(Uniform(ti.t0,ti.tF))

function compute_in_timesθ(mat::Matrix{Float},θ::Real;mat0=zeros(size(mat,1)))
  mat_prev = hcat(mat0,mat[:,1:end-1])
  θ*mat + (1-θ)*mat_prev
end

struct Nonaffine <: OperatorType end
abstract type ParamOperator{Top,Ttr} end
abstract type ParamLinOperator{Top} <: ParamOperator{Top,nothing} end
abstract type ParamBilinOperator{Top,Ttr} <: ParamOperator{Top,Ttr} end
abstract type ParamLiftOperator{Top} <: ParamLinOperator{Top} end

struct ParamSteadyLinOperator{Top} <: ParamLinOperator{Top}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  test::SingleFieldFESpace
end

struct ParamUnsteadyLinOperator{Top} <: ParamLinOperator{Top}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tinfo::TimeInfo
  test::SingleFieldFESpace
end

struct ParamSteadyBilinOperator{Top,Ttr} <: ParamBilinOperator{Top,Ttr}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  trial::Ttr
  test::SingleFieldFESpace
end

struct ParamUnsteadyBilinOperator{Top,Ttr} <: ParamBilinOperator{Top,Ttr}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tinfo::TimeInfo
  trial::Ttr
  test::SingleFieldFESpace
end

struct ParamSteadyLiftOperator{Top} <: ParamLiftOperator{Top}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  trial::ParamTrialFESpace
  test::SingleFieldFESpace
end

struct ParamUnsteadyLiftOperator{Top} <: ParamLiftOperator{Top}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tinfo::TimeInfo
  trial::ParamTransientTrialFESpace
  test::SingleFieldFESpace
end

get_id(op::ParamOperator) = op.id

get_param_function(op::ParamOperator) = op.a

get_fe_function(op::ParamOperator) = op.afe

Gridap.ODEs.TransientFETools.get_test(op::ParamOperator) = op.test

Gridap.FESpaces.get_trial(op::ParamOperator) = op.trial

get_pspace(op::ParamOperator) = op.pspace

get_time_info(op::ParamOperator) = op.tinfo

get_dt(op::ParamOperator) = get_dt(get_time_info(op))

get_Nt(op::ParamOperator) = get_Nt(get_time_info(op))

get_θ(op::ParamOperator) = get_θ(get_time_info(op))

get_timesθ(op::ParamOperator) = get_timesθ(get_time_info(op))

get_phys_quad_points(op::ParamOperator) = get_phys_quad_points(get_test(op))

realization(op::ParamOperator) = realization(get_pspace(op))

function Gridap.FESpaces.get_cell_dof_ids(
  op::ParamOperator,
  trian::Triangulation)
  collect(get_cell_dof_ids(get_test(op),trian))
end

function AffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,test::SingleFieldFESpace;id=:F)
  ParamSteadyLinOperator{Affine}(id,a,afe,pspace,test)
end

function NonaffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,test::SingleFieldFESpace;id=:F)
  ParamSteadyLinOperator{Nonaffine}(id,a,afe,pspace,test)
end

function NonlinearParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,test::SingleFieldFESpace;id=:F)
  ParamSteadyLinOperator{Nonlinear}(id,a,afe,pspace,test)
end

function AffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  test::SingleFieldFESpace;id=:F)
  ParamUnsteadyLinOperator{Affine}(id,a,afe,pspace,time_info,test)
end

function NonaffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  test::SingleFieldFESpace;id=:F)
  ParamUnsteadyLinOperator{Nonaffine}(id,a,afe,pspace,time_info,test)
end

function NonlinearParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  test::SingleFieldFESpace;id=:F)
  ParamUnsteadyLinOperator{Nonlinear}(id,a,afe,pspace,time_info,test)
end

function AffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamSteadyBilinOperator{Affine,Ttr}(id,a,afe,pspace,trial,test)
end

function NonaffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamSteadyBilinOperator{Nonaffine,Ttr}(id,a,afe,pspace,trial,test)
end

function NonlinearParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamSteadyBilinOperator{Nonlinear,Ttr}(id,a,afe,pspace,trial,test)
end

function AffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Affine,Ttr}(id,a,afe,pspace,time_info,trial,test)
end

function NonaffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Nonaffine,Ttr}(id,a,afe,pspace,time_info,trial,test)
end

function NonlinearParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Nonlinear,Ttr}(id,a,afe,pspace,time_info,trial,test)
end

function ParamLiftOperator(op::ParamSteadyBilinOperator{Affine,Ttr}) where Ttr
  id = get_id(op)*:_lift
  ParamSteadyLiftOperator{Nonaffine}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_trial(op),get_test(op))
end

function ParamLiftOperator(op::ParamSteadyBilinOperator{Top,Ttr}) where {Top,Ttr}
  id = get_id(op)*:_lift
  ParamSteadyLiftOperator{Top}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_trial(op),get_test(op))
end

function ParamLiftOperator(op::ParamUnsteadyBilinOperator{Affine,Ttr}) where Ttr
  id = get_id(op)*:_lift
  ParamUnsteadyLiftOperator{Nonaffine}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_time_info(op),get_trial(op),get_test(op))
end

function ParamLiftOperator(op::ParamUnsteadyBilinOperator{Top,Ttr}) where {Top,Ttr}
  id = get_id(op)*:_lift
  ParamUnsteadyLiftOperator{Top}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_time_info(op),get_trial(op),get_test(op))
end

function Gridap.FESpaces.assemble_vector(op::ParamSteadyLinOperator)
  afe = get_fe_function(op)
  test = get_test(op)
  μ -> assemble_vector(afe(μ),test)
end

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLinOperator)
  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  test = get_test(op)
  V(μ,tθ) = assemble_vector(afe(μ,tθ),test)
  μ -> Matrix([V(μ,tθ) for tθ = timesθ])
end

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLinOperator,t::Real)
  afe = get_fe_function(op)
  test = get_test(op)
  μ -> assemble_vector(afe(μ,t),test)
end

function Gridap.FESpaces.assemble_matrix(op::ParamSteadyBilinOperator)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ -> assemble_matrix(afe(μ),trial(μ),test)
end

function Gridap.FESpaces.assemble_matrix(op::ParamUnsteadyBilinOperator)
  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  M(μ,tθ) = assemble_matrix(afe(μ,tθ),trial(μ,tθ),test)
  μ -> [M(μ,tθ) for tθ = timesθ]
end

function Gridap.FESpaces.assemble_matrix(op::ParamUnsteadyBilinOperator,t::Real)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ -> assemble_matrix(afe(μ,t),trial(μ,t),test)
end

function Gridap.FESpaces.assemble_matrix(
  op::ParamSteadyBilinOperator{Nonlinear,Ttr}) where Ttr

  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  (μ,u) -> assemble_matrix(afe(u),trial(μ),test)
end

function Gridap.FESpaces.assemble_matrix(
  op::ParamUnsteadyBilinOperator{Nonlinear,Ttr}) where Ttr

  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  M(μ,t,u) = assemble_matrix(afe(u(t)),trial(μ,t),test)
  (μ,u) -> [M(μ,tθ,u) for tθ = timesθ]
end

function Gridap.FESpaces.assemble_matrix(
  op::ParamUnsteadyBilinOperator{Nonlinear,Ttr},
  t::Real) where Ttr

  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  (μ,u) -> assemble_matrix(afe(u),trial(μ,t),test)
end

function Gridap.FESpaces.assemble_vector(op::ParamSteadyLiftOperator)
  afe = get_fe_function(op)
  dir = get_dirichlet_function(op)
  test = get_test(op)
  lift(μ) = assemble_vector(v->afe(μ,dir(μ),v),test)
  lift
end

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLiftOperator)
  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  dir = get_dirichlet_function(op)
  test = get_test(op)
  lift(μ,t) = assemble_vector(v->afe(μ,t,dir(μ,t),v),test)
  μ -> Matrix([lift(μ,tθ) for tθ = timesθ])
end

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLiftOperator,t::Real)
  afe = get_fe_function(op)
  dir = get_dirichlet_function(op)
  test = get_test(op)
  lift(μ) = assemble_vector(v->afe(μ,t,dir(μ,t),v),test)
  lift
end

function Gridap.FESpaces.assemble_vector(op::ParamSteadyLiftOperator{Nonlinear})
  afe = get_fe_function(op)
  dir = get_dirichlet_function(op)
  test = get_test(op)
  (μ,u) -> assemble_vector(v->afe(u,dir(μ),v),test)
end

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLiftOperator{Nonlinear})
  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  dir = get_dirichlet_function(op)
  test = get_test(op)
  lift(μ,t,u) = assemble_vector(v->afe(u(t),dir(μ,t),v),test)
  (μ,u) -> [lift(μ,tθ,u) for tθ = timesθ]
end

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLiftOperator{Nonlinear},t::Real)
  afe = get_fe_function(op)
  dir = get_dirichlet_function(op)
  test = get_test(op)
  (μ,u) -> assemble_vector(v->afe(u,dir(μ,t),v),test)
end

function get_dirichlet_function(::Val,trial::ParamTrialFESpace)
  dir = trial.dirichlet_μt
  μ -> interpolate_dirichlet(dir(μ),trial(μ))
end

function get_dirichlet_function(::Val{false},trial::ParamTransientTrialFESpace)
  dir = trial.dirichlet_μt
  (μ,t) -> interpolate_dirichlet(dir(μ,t),trial(μ,t))
end

function get_dirichlet_function(::Val{true},trial::ParamTransientTrialFESpace)
  ddir = ∂t(trial.dirichlet_μt)
  (μ,t) -> interpolate_dirichlet(ddir(μ,t),trial(μ,t))
end

function get_dirichlet_function(op::ParamOperator)
  id = get_id(op)
  trial = get_trial(op)
  get_dirichlet_function(Val(id ∈ (:M,:M_lift)),trial)
end

function assemble_functional_variable(op::ParamLinOperator)
  afe = get_fe_function(op)
  test = get_test(op)
  fun -> assemble_vector(v->afe(fun,v),test)
end

function assemble_functional_variable(op::ParamSteadyBilinOperator)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  (fun,μ) -> assemble_matrix((u,v)->afe(fun,u,v),trial(μ),test)
end

function assemble_functional_variable(op::ParamUnsteadyBilinOperator)
  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  M(fun,μ,t) = assemble_matrix((u,v)->afe(fun,u,v),trial(μ,t),test)
  (fun,μ) -> [M(fun,μ,tθ) for tθ = timesθ]
end

function assemble_functional_variable(op::ParamSteadyLiftOperator)
  afe = get_fe_function(op)
  dir = get_dirichlet_function(op)
  test = get_test(op)
  (fun,μ) -> assemble_vector(v->afe(fun,dir(μ),v),test)
end

function assemble_functional_variable(op::ParamUnsteadyLiftOperator)
  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  dir = get_dirichlet_function(op)
  test = get_test(op)
  lift(fun,μ,t) = assemble_vector(v->afe(fun,dir(μ,t),v),test)
  (fun,μ) -> Matrix([lift(fun,μ,tθ) for tθ = timesθ])
end

function assemble_affine_variable(op::ParamSteadyLinOperator)
  assemble_vector(op)(realization(op))
end

function assemble_affine_variable(op::ParamUnsteadyLinOperator)
  assemble_vector(op,realization(op.tinfo))(realization(op))
end

function assemble_affine_variable(op::ParamSteadyBilinOperator)
  assemble_matrix(op)(realization(op))
end

function assemble_affine_variable(op::ParamUnsteadyBilinOperator)
  assemble_matrix(op,realization(op.tinfo))(realization(op))
end
