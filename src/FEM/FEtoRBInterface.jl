abstract type MySpaces end

function dirichlet_dofs_on_full_trian(space,space_no_bc)

  cell_dof_ids = get_cell_dof_ids(space)
  cell_dof_ids_no_bc = get_cell_dof_ids(space_no_bc)

  dirichlet_dofs = zeros(Int,space.ndirichlet)
  for cell = eachindex(cell_dof_ids)
    for (ids,ids_no_bc) in zip(cell_dof_ids[cell],cell_dof_ids_no_bc[cell])
      if ids<0
        dirichlet_dofs[abs(ids)]=ids_no_bc
      end
    end
  end

  dirichlet_dofs
end

struct MyTests <: MySpaces
  test::UnconstrainedFESpace
  test_no_bc::UnconstrainedFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTests(
    test::UnconstrainedFESpace,
    test_no_bc::UnconstrainedFESpace)

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(test,test_no_bc)
    new(test,test_no_bc,ddofs_on_full_trian)
  end
end

function MyTests(model,reffe;kwargs...)
  test = TestFESpace(model,reffe;kwargs...)
  test_no_bc = FESpace(model,reffe)
  MyTests(test,test_no_bc)
end

function MyTrial(test::UnconstrainedFESpace)
  HomogeneousTrialFESpace(test)
end

function MyTrial(
  test::UnconstrainedFESpace,
  g::Function,
  ::Val{true})
  ParamTrialFESpace(test,g)
end

function MyTrial(
  test::UnconstrainedFESpace,
  g::Function,
  ::Val{false})
  ParamTransientTrialFESpace(test,g)
end

function MyTrial(
  test::UnconstrainedFESpace,
  g::Function,
  ptype::ProblemType)
  MyTrial(test,g,issteady(ptype))
end

struct MyTrials{Ttr} <: MySpaces
  trial::Ttr
  trial_no_bc::UnconstrainedFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTrials(
    trial::Ttr,
    trial_no_bc::UnconstrainedFESpace) where Ttr

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(trial.space,trial_no_bc)
    new{Ttr}(trial,trial_no_bc,ddofs_on_full_trian)
  end
end

function MyTrials(tests::MyTests,args...)
  trial = MyTrial(tests.test,args...)
  trial_no_bc = TrialFESpace(tests.test_no_bc)
  MyTrials(trial,trial_no_bc)
end

function ParamAffineFEOperator(a::Function,b::Function,
  pspace::ParamSpace,trial::MyTrials,test::MyTests)
  ParamAffineFEOperator(a,b,pspace,get_trial(trial),get_test(test))
end

function ParamFEOperator(res::Function,jac::Function,
  pspace::ParamSpace,trial::MyTrials,test::MyTests)
  ParamFEOperator(res,jac,pspace,get_trial(trial),get_test(test))
end

function ParamMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamMultiFieldFESpace([get_trial(first(spaces)),get_trial(last(spaces))])
end

function ParamMultiFieldFESpace(spaces::Vector{MyTests})
  ParamMultiFieldFESpace([get_test(first(spaces)),get_test(last(spaces))])
end

function ParamTransientAffineFEOperator(m::Function,a::Function,b::Function,
  pspace::ParamSpace,trial::MyTrials,test::MyTests)
  ParamTransientAffineFEOperator(m,a,b,pspace,get_trial(trial),get_test(test))
end

function ParamTransientFEOperator(res::Function,jac::Function,jac_t::Function,
  pspace::ParamSpace,trial::MyTrials,test::MyTests)
  ParamTransientFEOperator(res,jac,jac_t,pspace,get_trial(trial),get_test(test))
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamTransientMultiFieldFESpace([get_trial(first(spaces)),get_trial(last(spaces))])
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTests})
  ParamTransientMultiFieldFESpace([get_test(first(spaces)),get_test(last(spaces))])
end

function free_dofs_on_full_trian(tests::MyTests)
  nfree_on_full_trian = tests.test_no_bc.nfree
  setdiff(collect(1:nfree_on_full_trian),tests.ddofs_on_full_trian)
end

function free_dofs_on_full_trian(trials::MyTrials)
  nfree_on_full_trian = trials.trial_no_bc.nfree
  setdiff(collect(1:nfree_on_full_trian),trials.ddofs_on_full_trian)
end

function get_fd_dofs(tests::MyTests,trials::MyTrials)
  fdofs_test = free_dofs_on_full_trian(tests)
  fdofs_trial = free_dofs_on_full_trian(trials)
  ddofs = trials.ddofs_on_full_trian
  (fdofs_test,fdofs_trial),ddofs
end

function Gridap.get_background_model(test::UnconstrainedFESpace)
  get_background_model(get_triangulation(test))
end

function get_dimension(test::UnconstrainedFESpace)
  model = get_background_model(test)
  maximum(model.grid.reffes[1].reffe.polytope.dface.dims)
end

function Gridap.FESpaces.get_order(test::UnconstrainedFESpace)
  basis = get_fe_basis(test)
  first(basis.cell_basis.values[1].fields.orders)
end

Gridap.FESpaces.get_test(tests::MyTests) = tests.test
Gridap.FESpaces.get_trial(trials::MyTrials) = trials.trial
get_test_no_bc(tests::MyTests) = tests.test_no_bc
get_trial_no_bc(trials::MyTrials) = trials.trial_no_bc
get_degree(order::Int,c=2) = c*order
get_degree(test::UnconstrainedFESpace,c=2) = get_degree(Gridap.FESpaces.get_order(test),c)
realization(fes::MySpaces) = FEFunction(fes.test,rand(num_free_dofs(fes.test)))

function get_cell_quadrature(test::UnconstrainedFESpace)
  CellQuadrature(get_triangulation(test),get_degree(test))
end

struct LagrangianQuadFESpace
  test::UnconstrainedFESpace
end

function LagrangianQuadFESpace(model::DiscreteModel,order::Int)
  reffe_quad = Gridap.ReferenceFE(lagrangian_quad,Float,order)
  test = TestFESpace(model,reffe_quad,conformity=:L2)
  LagrangianQuadFESpace(test)
end

function LagrangianQuadFESpace(test::UnconstrainedFESpace)
  model = get_background_model(test)
  order = Gridap.FESpaces.get_order(test)
  LagrangianQuadFESpace(model,order)
end

function LagrangianQuadFESpace(tests::MyTests)
  LagrangianQuadFESpace(get_test(tests))
end

function get_phys_quad_points(test::UnconstrainedFESpace)
  trian = get_triangulation(test)
  phys_map = get_cell_map(trian)
  cell_quad = get_cell_quadrature(test)
  cell_points = get_data(get_cell_points(cell_quad))
  map(Gridap.evaluate,phys_map,cell_points)
end

function get_phys_quad_points(tests::MyTests)
  get_phys_quad_points(get_test(tests))
end

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

struct Nonaffine <: OperatorType end
abstract type ParamVarOperator{Top,Ttr} end
abstract type ParamLinOperator{Top} <: ParamVarOperator{Top,nothing} end
abstract type ParamBilinOperator{Top,Ttr} <: ParamVarOperator{Top,Ttr} end

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

abstract type ParamLiftingOperator{Top,Ttr} <: ParamLinOperator{Top} end

struct ParamSteadyLiftingOperator{Top,Ttr} <: ParamLiftingOperator{Top,Ttr}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  trials::MyTrials{Ttr}
  tests::MyTests
end

struct ParamUnsteadyLiftingOperator{Top,Ttr} <: ParamLiftingOperator{Top,Ttr}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tinfo::TimeInfo
  trials::MyTrials{Ttr}
  tests::MyTests
end

get_id(op::ParamVarOperator) = op.id
get_param_function(op::ParamVarOperator) = op.a
get_fe_function(op::ParamVarOperator) = op.afe
Gridap.ODEs.TransientFETools.get_test(op::ParamVarOperator) = op.tests.test
Gridap.FESpaces.get_trial(op::ParamBilinOperator) = op.trials.trial
get_tests(op::ParamVarOperator) = op.tests
get_trials(op::ParamBilinOperator) = op.trials
get_test_no_bc(op::ParamVarOperator) = op.tests.test_no_bc
get_trial_no_bc(op::ParamBilinOperator) = op.trials.trial_no_bc
get_pspace(op::ParamVarOperator) = op.pspace

get_time_info(op::ParamVarOperator) = op.tinfo
get_dt(op::ParamVarOperator) = get_dt(get_time_info(op))
get_Nt(op::ParamVarOperator) = get_Nt(get_time_info(op))
get_θ(op::ParamVarOperator) = get_θ(get_time_info(op))
get_timesθ(op::ParamVarOperator) = get_timesθ(get_time_info(op))

Gridap.FESpaces.get_trial(op::ParamLiftingOperator) = op.trials.trial
get_trials(op::ParamLiftingOperator) = op.trials
get_trial_no_bc(op::ParamLiftingOperator) = op.trials.trial_no_bc

realization(op::ParamVarOperator) = realization(get_pspace(op))

function Gridap.FESpaces.get_cell_dof_ids(
  op::ParamVarOperator,
  trian::Triangulation)
  collect(get_cell_dof_ids(get_test(op),trian))
end

function AffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamSteadyLinOperator{Affine}(id,a,afe,pspace,tests)
end

function NonaffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamSteadyLinOperator{Nonaffine}(id,a,afe,pspace,tests)
end

function NonlinearParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamSteadyLinOperator{Nonlinear}(id,a,afe,pspace,tests)
end

function AffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  tests::MyTests;id=:F)
  ParamUnsteadyLinOperator{Affine}(id,a,afe,pspace,time_info,tests)
end

function NonaffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  tests::MyTests;id=:F)
  ParamUnsteadyLinOperator{Nonaffine}(id,a,afe,pspace,time_info,tests)
end

function NonlinearParamVarOperator(
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

function AffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamSteadyBilinOperator{Affine,Ttr}(id,a,afe,pspace,trials,tests)
end

function NonaffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamSteadyBilinOperator{Nonaffine,Ttr}(id,a,afe,pspace,trials,tests)
end

function NonlinearParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamSteadyBilinOperator{Nonlinear,Ttr}(id,a,afe,pspace,trials,tests)
end

function AffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Affine,Ttr}(id,a,afe,pspace,time_info,trials,tests)
end

function NonaffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Nonaffine,Ttr}(id,a,afe,pspace,time_info,trials,tests)
end

function NonlinearParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,time_info::TimeInfo,
  trials::MyTrials{Ttr},tests::MyTests;id=:A) where Ttr
  ParamUnsteadyBilinOperator{Nonlinear,Ttr}(id,a,afe,pspace,time_info,trials,tests)
end

function ParamLiftingOperator(::ParamBilinOperator{Top,<:TrialFESpace}) where Top
  error("No lifting operator associated to a trial space of type TrialFESpace")
end

function ParamLiftingOperator(op::ParamSteadyBilinOperator{Affine,Ttr}) where Ttr
  id = get_id(op)*:_lift
  ParamSteadyLiftingOperator{Nonaffine,Ttr}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_trials(op),get_tests(op))
end

function ParamLiftingOperator(op::ParamSteadyBilinOperator{Top,Ttr}) where {Top,Ttr}
  id = get_id(op)*:_lift
  ParamSteadyLiftingOperator{Top,Ttr}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_trials(op),get_tests(op))
end

function ParamLiftingOperator(op::ParamUnsteadyBilinOperator{Affine,Ttr}) where Ttr
  id = get_id(op)*:_lift
  ParamUnsteadyLiftingOperator{Nonaffine,Ttr}(id,get_param_function(op),
    get_fe_function(op),get_pspace(op),get_time_info(op),get_trials(op),get_tests(op))
end

function ParamLiftingOperator(op::ParamUnsteadyBilinOperator{Top,Ttr}) where {Top,Ttr}
  id = get_id(op)*:_lift
  ParamUnsteadyLiftingOperator{Top,Ttr}(id,get_param_function(op),
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

function Gridap.FESpaces.assemble_vector(op::ParamSteadyLiftingOperator)
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

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLiftingOperator)
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

function Gridap.FESpaces.assemble_vector(op::ParamUnsteadyLiftingOperator,t::Real)
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
  op::Union{ParamSteadyBilinOperator,ParamSteadyLiftingOperator})

  id = get_id(op)
  trial = get_trial(op)
  μ -> get_dirichlet_function(Val(id ∈ (:M,:M_lift)),trial)(μ)
end

function get_dirichlet_function(
  op::Union{ParamUnsteadyBilinOperator,ParamUnsteadyLiftingOperator})

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

get_nsnap(v::AbstractVector) = length(v)
get_nsnap(m::AbstractMatrix) = size(m)[2]

mutable struct Snapshots{T}
  id::Symbol
  snap::AbstractArray{T}
  nsnap::Int
end

function Snapshots(id::Symbol,snap::AbstractArray{T},nsnap::Int) where T
  Snapshots{T}(id,snap,nsnap)
end

function Snapshots(id::Symbol,snap::AbstractArray)
  nsnap = get_nsnap(snap)
  Snapshots(id,snap,nsnap)
end

function Snapshots(id::Symbol,blocks::Vector{<:AbstractArray})
  snap = Matrix(blocks)
  nsnap = get_nsnap(blocks)
  Snapshots(id,snap,nsnap)
end

function Snapshots(id::Symbol,snap::NTuple{N,AbstractArray}) where N
  Broadcasting(si->Snapshots(id,si))(snap)
end

function Base.getindex(s::Snapshots,idx::Int)
  Nt = get_Nt(s)
  snap_idx = s.snap[:,(idx-1)*Nt+1:idx*Nt]
  Snapshots(s.id,snap_idx,1)
end

function Base.getindex(s::Snapshots,idx::UnitRange{Int})
  Nt = get_Nt(s)
  snap(i) = getindex(s.snap,:,(i-1)*Nt+1:i*Nt)
  Snapshots(s.id,Matrix(snap.(idx)),length(idx))
end

get_id(s::Snapshots) = s.id
get_snap(s::Snapshots) = s.snap
get_nsnap(s::Snapshots) = s.nsnap

save(path::String,s::Snapshots) = save(joinpath(path,"$(s.id)"),s.snap)

function load_snap(path::String,id::Symbol,nsnap::Int)
  s = load(joinpath(path,"$(id)"))
  Snapshots(id,s,nsnap)
end

get_Nt(s::Snapshots) = get_Nt(get_snap(s),get_nsnap(s))
mode2_unfolding(s::Snapshots) = mode2_unfolding(get_snap(s),get_nsnap(s))
mode2_unfolding(s::NTuple{N,Snapshots}) where N = mode2_unfolding.(s)
POD(s::Snapshots,args...;kwargs...) = POD(s.snap,args...;kwargs...)
POD(s::NTuple{N,Snapshots},args...;kwargs...) where N = Broadcasting(si->POD(si,args...;kwargs...))(s)
