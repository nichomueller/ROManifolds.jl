struct ParamFunctions
  fun::Function
  fefun::Function
end

get_param_function(pf::ParamFunctions) = pf.fun
get_param_fefunction(pf::ParamFunctions) = pf.fefun

struct TimeInfo
  t0::Real
  tF::Real
  dt::Real
  θ::Real
end

get_dt(ti::TimeInfo) = ti.dt
get_Nt(ti::TimeInfo) = Int((ti.tF-ti.t0)/ti.dt)
get_θ(ti::TimeInfo) = ti.θ
get_times(ti::TimeInfo) = collect(ti.t0:ti.dt:ti.tF-ti.dt).+ti.dt*ti.θ
realization(ti::TimeInfo) = rand(Uniform(ti.t0,ti.tF))

function compute_in_times(mat::Matrix{Float},θ::Real;mat0=zeros(size(mat,1)))
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
  pfun::ParamFunctions
  pspace::ParamSpace
  test::SingleFieldFESpace
end

struct ParamUnsteadyLinOperator{Top} <: ParamLinOperator{Top}
  id::Symbol
  pfun::ParamFunctions
  pspace::ParamSpace
  tinfo::TimeInfo
  test::SingleFieldFESpace
end

struct ParamSteadyBilinOperator{Top,Ttr} <: ParamBilinOperator{Top,Ttr}
  id::Symbol
  pfun::ParamFunctions
  pspace::ParamSpace
  trial::Ttr
  test::SingleFieldFESpace
end

struct ParamUnsteadyBilinOperator{Top,Ttr} <: ParamBilinOperator{Top,Ttr}
  id::Symbol
  pfun::ParamFunctions
  pspace::ParamSpace
  tinfo::TimeInfo
  trial::Ttr
  test::SingleFieldFESpace
end

struct ParamSteadyLiftOperator{Top} <: ParamLiftOperator{Top}
  id::Symbol
  pfun::ParamFunctions
  pspace::ParamSpace
  trial::ParamTrialFESpace
  test::SingleFieldFESpace
end

struct ParamUnsteadyLiftOperator{Top} <: ParamLiftOperator{Top}
  id::Symbol
  pfun::ParamFunctions
  pspace::ParamSpace
  tinfo::TimeInfo
  trial::ParamTransientTrialFESpace
  test::SingleFieldFESpace
end

const ParamSteadyOperator{Top,Ttr} =
  Union{ParamSteadyLinOperator{Top},ParamSteadyBilinOperator{Top,Ttr},ParamSteadyLiftOperator{Top}}

const ParamUnsteadyOperator{Top,Ttr} =
  Union{ParamUnsteadyLinOperator{Top},ParamUnsteadyBilinOperator{Top,Ttr},ParamUnsteadyLiftOperator{Top}}

get_id(op::ParamOperator) = op.id

get_param_functions(op::ParamOperator) = op.pfun

get_param_function(op::ParamOperator) = get_param_function(op.pfun)

get_param_fefunction(op::ParamOperator) = get_param_fefunction(op.pfun)

Gridap.ODEs.TransientFETools.get_test(op::ParamOperator) = op.test

Gridap.FESpaces.get_trial(op::ParamOperator) = op.trial

get_pspace(op::ParamOperator) = op.pspace

get_time_info(op::ParamUnsteadyOperator) = op.tinfo

get_dt(op::ParamUnsteadyOperator) = get_dt(get_time_info(op))

get_Nt(op::ParamUnsteadyOperator) = get_Nt(get_time_info(op))

get_θ(op::ParamUnsteadyOperator) = get_θ(get_time_info(op))

get_times(::ParamSteadyOperator) = fill(nothing,1)

get_times(op::ParamUnsteadyOperator) = get_times(get_time_info(op))

get_phys_quad_points(op::ParamOperator) = get_phys_quad_points(get_test(op))

get_dimension(op::ParamOperator) = get_dimension(get_test(op))

realization(op::ParamOperator) = realization(get_pspace(op))

realization_trial(op::ParamSteadyOperator) = get_trial(op)(realization(op))

realization_trial(op::ParamUnsteadyOperator) = get_trial(op)(realization(op),realization(get_time_info(op)))

function Gridap.FESpaces.get_cell_dof_ids(
  op::ParamOperator,
  trian::Triangulation)
  collect(get_cell_dof_ids(get_test(op),trian))
end

function AffineParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,test::SingleFieldFESpace;id=:F)
  ParamSteadyLinOperator{Affine}(id,pfun,pspace,test)
end

function NonaffineParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,test::SingleFieldFESpace;id=:F)
  ParamSteadyLinOperator{Nonaffine}(id,pfun,pspace,test)
end

function NonlinearParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,test::SingleFieldFESpace;id=:F)
  ParamSteadyLinOperator{Nonlinear}(id,pfun,pspace,test)
end

function AffineParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,time_info::TimeInfo,
  test::SingleFieldFESpace;id=:F)
  ParamUnsteadyLinOperator{Affine}(id,pfun,pspace,time_info,test)
end

function NonaffineParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,time_info::TimeInfo,
  test::SingleFieldFESpace;id=:F)
  ParamUnsteadyLinOperator{Nonaffine}(id,pfun,pspace,time_info,test)
end

function NonlinearParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,time_info::TimeInfo,
  test::SingleFieldFESpace;id=:F)
  ParamUnsteadyLinOperator{Nonlinear}(id,pfun,pspace,time_info,test)
end

function AffineParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamSteadyBilinOperator{Affine,Ttr}(id,pfun,pspace,trial,test)
end

function NonaffineParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamSteadyBilinOperator{Nonaffine,Ttr}(id,pfun,pspace,trial,test)
end

function NonlinearParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamSteadyBilinOperator{Nonlinear,Ttr}(id,pfun,pspace,trial,test)
end

function AffineParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,time_info::TimeInfo,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Affine,Ttr}(id,pfun,pspace,time_info,trial,test)
end

function NonaffineParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,time_info::TimeInfo,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Nonaffine,Ttr}(id,pfun,pspace,time_info,trial,test)
end

function NonlinearParamOperator(
  pfun::ParamFunctions,pspace::ParamSpace,time_info::TimeInfo,
  trial::Ttr,test::SingleFieldFESpace;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Nonlinear,Ttr}(id,pfun,pspace,time_info,trial,test)
end

function ParamLiftOperator(op::ParamSteadyBilinOperator{Affine,Ttr}) where Ttr
  id = get_id(op)*:_lift
  ParamSteadyLiftOperator{Nonaffine}(id,get_param_functions(op),
    get_pspace(op),get_trial(op),get_test(op))
end

function ParamLiftOperator(op::ParamSteadyBilinOperator{Top,Ttr}) where {Top,Ttr}
  id = get_id(op)*:_lift
  ParamSteadyLiftOperator{Top}(id,get_param_functions(op),
    get_pspace(op),get_trial(op),get_test(op))
end

function ParamLiftOperator(op::ParamUnsteadyBilinOperator{Affine,Ttr}) where Ttr
  id = get_id(op)*:_lift
  ParamUnsteadyLiftOperator{Nonaffine}(id,get_param_functions(op),
    get_pspace(op),get_time_info(op),get_trial(op),get_test(op))
end

function ParamLiftOperator(op::ParamUnsteadyBilinOperator{Top,Ttr}) where {Top,Ttr}
  id = get_id(op)*:_lift
  ParamUnsteadyLiftOperator{Top}(id,get_param_functions(op),
    get_pspace(op),get_time_info(op),get_trial(op),get_test(op))
end

function assemble_affine_quantity(op::ParamOperator)
  assemble_affine_quantity(unpack_for_assembly(op)...,realization(op),first(get_times(op)))
end

function assemble_affine_quantity(fefun::Function,test::FESpace,args...)
  assemble_vector(fefun(x->1),test)
end

function assemble_affine_quantity(fefun::Function,test::FESpace,trial::ParamTrialFESpace,μ::Param,args...)
  assemble_matrix(fefun(x->1),trial(μ),test)
end

function assemble_affine_quantity(fefun::Function,test::FESpace,trial::ParamTransientTrialFESpace,μ::Param,t::Real)
  assemble_matrix(fefun(x->1),trial(μ,t),test)
end

function get_dirichlet_function(::Val,trial::ParamTrialFESpace)
  μ -> zero(trial(μ))
end

function get_dirichlet_function(::Val{false},trial::ParamTransientTrialFESpace)
  (μ,t) -> zero(trial(μ,t))
end

function get_dirichlet_function(::Val{true},trial::ParamTransientTrialFESpace)
  (μ,t) -> zero(∂t(trial)(μ,t))
end

function get_dirichlet_function(op::ParamOperator)
  id = get_id(op)
  trial = get_trial(op)
  get_dirichlet_function(Val(id ∈ (:M,:M_lift)),trial)
end
