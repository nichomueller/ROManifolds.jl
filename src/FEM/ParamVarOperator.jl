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

struct ParamSteadyLinOperator{Top} <: ParamLinOperator{Top}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tests::MyTests
end

struct ParamUnsteadyLinOperator{Top} <: ParamLinOperator{Top}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tinfo::TimeInfo
  tests::MyTests
end

struct ParamSteadyBilinOperator{Top,Ttr} <: ParamBilinOperator{Top,Ttr}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  trials::MyTrials{Ttr}
  tests::MyTests
end

struct ParamUnsteadyBilinOperator{Top,Ttr} <: ParamBilinOperator{Top,Ttr}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tinfo::TimeInfo
  trials::MyTrials{Ttr}
  tests::MyTests
end

abstract type ParamLiftOperator{Top,Ttr} <: ParamLinOperator{Top} end

struct ParamSteadyLiftOperator{Top,Ttr} <: ParamLiftOperator{Top,Ttr}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  trials::MyTrials{Ttr}
  tests::MyTests
end

struct ParamUnsteadyLiftOperator{Top,Ttr} <: ParamLiftOperator{Top,Ttr}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tinfo::TimeInfo
  trials::MyTrials{Ttr}
  tests::MyTests
end

get_id(op::ParamOperator) = op.id
get_param_function(op::ParamOperator) = op.a
get_fe_function(op::ParamOperator) = op.afe
Gridap.ODEs.TransientFETools.get_test(op::ParamOperator) = op.tests.test
Gridap.FESpaces.get_trial(op::ParamBilinOperator) = op.trials.trial
get_tests(op::ParamOperator) = op.tests
get_trials(op::ParamBilinOperator) = op.trials
get_test_no_bc(op::ParamOperator) = op.tests.test_no_bc
get_trial_no_bc(op::ParamBilinOperator) = op.trials.trial_no_bc
get_pspace(op::ParamOperator) = op.pspace

get_time_info(op::ParamOperator) = op.tinfo
get_dt(op::ParamOperator) = get_dt(get_time_info(op))
get_Nt(op::ParamOperator) = get_Nt(get_time_info(op))
get_θ(op::ParamOperator) = get_θ(get_time_info(op))
get_timesθ(op::ParamOperator) = get_timesθ(get_time_info(op))

Gridap.FESpaces.get_trial(op::ParamLiftOperator) = op.trials.trial
get_trials(op::ParamLiftOperator) = op.trials
get_trial_no_bc(op::ParamLiftOperator) = op.trials.trial_no_bc
get_phys_quad_points(op::ParamOperator) = get_phys_quad_points(get_test(op))

realization(op::ParamOperator) = realization(get_pspace(op))

function Gridap.FESpaces.get_cell_dof_ids(
  op::ParamOperator,
  trian::Triangulation)
  collect(get_cell_dof_ids(get_test(op),trian))
end

function AffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamSteadyLinOperator{Affine}(id,a,afe,pspace,tests)
end

function NonaffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamSteadyLinOperator{Nonaffine}(id,a,afe,pspace,tests)
end

function NonlinearParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamSteadyLinOperator{Nonlinear}(id,a,afe,pspace,tests)
end

function AffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  tests::MyTests;id=:F)
  ParamUnsteadyLinOperator{Affine}(id,a,afe,pspace,time_info,tests)
end

function NonaffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  tests::MyTests;id=:F)
  ParamUnsteadyLinOperator{Nonaffine}(id,a,afe,pspace,time_info,tests)
end

function NonlinearParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  tests::MyTests;id=:F)
  ParamUnsteadyLinOperator{Nonlinear}(id,a,afe,pspace,time_info,tests)
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

function AffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamSteadyBilinOperator{Affine,Ttr}(id,a,afe,pspace,trials,tests)
end

function NonaffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamSteadyBilinOperator{Nonaffine,Ttr}(id,a,afe,pspace,trials,tests)
end

function NonlinearParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamSteadyBilinOperator{Nonlinear,Ttr}(id,a,afe,pspace,trials,tests)
end

function AffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Affine,Ttr}(id,a,afe,pspace,time_info,trials,tests)
end

function NonaffineParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Nonaffine,Ttr}(id,a,afe,pspace,time_info,trials,tests)
end

function NonlinearParamOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Nonlinear,Ttr}(id,a,afe,pspace,time_info,trials,tests)
end

function ParamLiftOperator(::ParamBilinOperator{Top,<:TrialFESpace}) where Top
  error("No lifting operator associated to a trial space of type TrialFESpace")
end

function ParamLiftOperator(op::ParamSteadyBilinOperator{Affine,Ttr}) where Ttr
  id = get_id(op)*:_lift
  ParamSteadyLiftOperator{Nonaffine,Ttr}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_trials(op),get_tests(op))
end

function ParamLiftOperator(op::ParamSteadyBilinOperator{Top,Ttr}) where {Top,Ttr}
  id = get_id(op)*:_lift
  ParamSteadyLiftOperator{Top,Ttr}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_trials(op),get_tests(op))
end

function ParamLiftOperator(op::ParamUnsteadyBilinOperator{Affine,Ttr}) where Ttr
  id = get_id(op)*:_lift
  ParamUnsteadyLiftOperator{Nonaffine,Ttr}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_time_info(op),get_trials(op),get_tests(op))
end

function ParamLiftOperator(op::ParamUnsteadyBilinOperator{Top,Ttr}) where {Top,Ttr}
  id = get_id(op)*:_lift
  ParamUnsteadyLiftOperator{Top,Ttr}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_time_info(op),get_trials(op),get_tests(op))
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
  op::ParamSteadyBilinOperator{Top,<:TrialFESpace}) where Top

  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ -> assemble_matrix(afe(μ),trial,test)
end

function Gridap.FESpaces.assemble_matrix(
  op::ParamUnsteadyBilinOperator{Top,<:TrialFESpace}) where Top

  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  t = realization(op.tinfo)
  μ -> assemble_matrix(afe(μ,t),trial,test)
end

function Gridap.FESpaces.assemble_matrix(
  op::ParamUnsteadyBilinOperator{Top,<:TrialFESpace},
  t::Real) where Top

  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ -> assemble_matrix(afe(μ,t),trial,test)
end

function Gridap.FESpaces.assemble_matrix(
  op::ParamSteadyBilinOperator{Nonlinear,Ttr}) where Ttr

  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ = realization(op)
  u -> assemble_matrix(afe(u),trial(μ),test)
end

function Gridap.FESpaces.assemble_matrix(
  op::ParamUnsteadyBilinOperator{Nonlinear,Ttr}) where Ttr

  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ = realization(op)
  t = realization(op.tinfo)
  u -> assemble_matrix(afe(u),trial(μ,t),test)
end

function Gridap.FESpaces.assemble_vector(op::ParamSteadyLiftOperator)
  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ),trial_no_bc,test_no_bc)
  dir = get_dirichlet_function(op)
  lift(μ) = A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)

  lift
end

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLiftOperator)
  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  A_no_bc(μ,tθ) = assemble_matrix(afe(μ,tθ),trial_no_bc,test_no_bc)
  dir = get_dirichlet_function(op)
  lift(μ,tθ) = A_no_bc(μ,tθ)[fdofs_test,ddofs]*dir(μ,tθ)
  lift(μ) = Matrix([lift(μ,tθ) for tθ = timesθ])

  lift
end

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLiftOperator,t::Real)
  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ,t),trial_no_bc,test_no_bc)
  dir = get_dirichlet_function(op)
  lift(μ) = A_no_bc(μ)[fdofs_test,ddofs]*dir(μ,t)

  lift
end

function Gridap.FESpaces.assemble_vector(
  op::ParamSteadyLinOperator{TrialFESpace})

  afe = get_fe_function(op)
  test = get_test(op)
  μ -> assemble_vector(afe(μ),test)
end

function Gridap.FESpaces.assemble_vector(
  op::ParamUnsteadyLinOperator{TrialFESpace})

  afe = get_fe_function(op)
  test = get_test(op)
  t = realization(op.tinfo)
  μ -> assemble_vector(afe(μ,t),test)
end

function Gridap.FESpaces.assemble_vector(
  op::ParamUnsteadyLinOperator{TrialFESpace},
  t::Real)

  afe = get_fe_function(op)
  test = get_test(op)
  μ -> assemble_vector(afe(μ,t),test)
end

function get_dirichlet_function(::Val,trial::ParamTrialFESpace)
  μ -> trial(μ).dirichlet_values
end

function get_dirichlet_function(::Val{false},trial::ParamTransientTrialFESpace)
  (μ,t) -> trial(μ,t).dirichlet_values
end

function get_dirichlet_function(::Val{true},trial::ParamTransientTrialFESpace)
  dg = ∂t(trial.dirichlet_μt)
  dg_dirichlet(μ,t) = interpolate_dirichlet(dg(μ,t),trial(μ,t)).dirichlet_values
  dg_dirichlet
end

function get_dirichlet_function(
  op::Union{ParamSteadyBilinOperator,ParamSteadyLiftOperator})

  id = get_id(op)
  trial = get_trial(op)
  μ -> get_dirichlet_function(Val(id ∈ (:M,:M_lift)),trial)(μ)
end

function get_dirichlet_function(
  op::Union{ParamUnsteadyBilinOperator,ParamUnsteadyLiftOperator})

  id = get_id(op)
  trial = get_trial(op)
  (μ,t) -> get_dirichlet_function(Val(id ∈ (:M,:M_lift)),trial)(μ,t)
end

function assemble_matrix_and_lifting(op::ParamSteadyBilinOperator)
  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ),trial_no_bc,test_no_bc)
  A_bc(μ) = A_no_bc(μ)[fdofs_test,fdofs_trial]
  dir = get_dirichlet_function(op)
  lift(μ) = A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)

  A_bc,lift
end

function assemble_matrix_and_lifting(op::ParamUnsteadyBilinOperator)
  timesθ = get_timesθ(op)
  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ,tθ) = assemble_matrix(afe(μ,tθ),trial_no_bc,test_no_bc)
  A_bc(μ,tθ) = A_no_bc(μ,tθ)[fdofs_test,fdofs_trial]
  A_bc(μ) = [A_bc(μ,tθ) for tθ = timesθ]
  dir = get_dirichlet_function(op)
  lift(μ,tθ) = A_no_bc(μ,tθ)[fdofs_test,ddofs]*dir(μ,tθ)
  lift(μ) = Matrix([lift(μ,tθ) for tθ = timesθ])

  A_bc,lift
end

function assemble_matrix_and_lifting(op::ParamUnsteadyBilinOperator,t::Real)
  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ,t),trial_no_bc,test_no_bc)
  A_bc(μ) = A_no_bc(μ)[fdofs_test,fdofs_trial]
  dir = get_dirichlet_function(op)
  lift(μ) = A_no_bc(μ)[fdofs_test,ddofs]*dir(μ,t)

  A_bc,lift
end

function assemble_functional_vector(op::ParamLinOperator)
  afe = get_fe_function(op)
  test = get_test(op)
  fun -> assemble_vector(v->afe(fun,v),test)
end

function assemble_functional_vector(op::ParamUnsteadyLiftOperator)
  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  A_no_bc(fun) = assemble_matrix((u,v)->afe(fun,u,v),trial_no_bc,test_no_bc)
  dir = get_dirichlet_function(op)

  lift(fun,μ,tθ) = A_no_bc(fun)[fdofs_test,ddofs]*dir(μ,tθ)

  lift
end

function assemble_functional_matrix_and_lifting(op::ParamUnsteadyBilinOperator)
  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(fun) = assemble_matrix((u,v)->afe(fun,u,v),trial_no_bc,test_no_bc)
  A_bc(fun) = A_no_bc(fun)[fdofs_test,fdofs_trial]
  dir = get_dirichlet_function(op)
  lift(fun,μ,tθ) = A_no_bc(fun)[fdofs_test,ddofs]*dir(μ,tθ)

  A_bc,lift
end

function assemble_matrix_and_lifting(
  op::ParamBilinOperator{Top,<:TrialFESpace},
  args...) where Top
  error("Operator $(get_id(op)) has no lifting")
end

function assemble_matrix_and_lifting(
  op::ParamBilinOperator{Top,<:TrialFESpace},args...) where Top
  assemble_matrix(op,args...),nothing
end

function assemble_matrix_and_lifting(
  op::ParamSteadyBilinOperator{Nonlinear,Ttr}) where Ttr

  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(u) = assemble_matrix(afe(u),trial_no_bc,test_no_bc)
  A_bc(u) = A_no_bc(u)[fdofs_test,fdofs_trial]
  lift(u) = A_no_bc(u)[fdofs_test,ddofs]*u.dirichlet_values

  A_bc,lift
end

function assemble_matrix_and_lifting(
  op::ParamUnsteadyBilinOperator{Nonlinear,Ttr}) where Ttr

  afe = get_fe_function(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(u) = assemble_matrix(afe(u),trial_no_bc,test_no_bc)
  A_bc(u) = A_no_bc(u)[fdofs_test,fdofs_trial]
  lift(u) = A_no_bc(u)[fdofs_test,ddofs]*u.dirichlet_values

  A_bc,lift
end
