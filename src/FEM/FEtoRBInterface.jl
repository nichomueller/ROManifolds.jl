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

struct MyTrials{TT} <: MySpaces
  trial::TT
  trial_no_bc::UnconstrainedFESpace
  ddofs_on_full_trian::Vector{Int}

  function MyTrials(
    trial::TT,
    trial_no_bc::UnconstrainedFESpace) where TT

    ddofs_on_full_trian = dirichlet_dofs_on_full_trian(trial.space,trial_no_bc)
    new{TT}(trial,trial_no_bc,ddofs_on_full_trian)
  end
end

function MyTrials(tests::MyTests,args...)
  trial = MyTrial(tests.test,args...)
  trial_no_bc = TrialFESpace(tests.test_no_bc)
  MyTrials(trial,trial_no_bc)
end

function ParamMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamMultiFieldFESpace([first(spaces).trial,last(spaces).trial])
end

function ParamMultiFieldFESpace(spaces::Vector{MyTests})
  ParamMultiFieldFESpace([first(spaces).test,last(spaces).test])
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTrials})
  ParamTransientMultiFieldFESpace([first(spaces).trial,last(spaces).trial])
end

function ParamTransientMultiFieldFESpace(spaces::Vector{MyTests})
  ParamTransientMultiFieldFESpace([first(spaces).test,last(spaces).test])
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

function Gridap.FESpaces.get_order(test::UnconstrainedFESpace)
  Gridap.FESpaces.get_order(first(get_background_model(test).grid.reffes))
end

Gridap.FESpaces.get_test(tests::MyTests) = tests.test
Gridap.FESpaces.get_trial(trials::MyTrials) = trials.trial
get_test_no_bc(tests::MyTests) = tests.test_no_bc
get_trial_no_bc(trials::MyTrials) = trials.trial_no_bc
get_degree(order::Int,c=2) = c*order
get_degree(test::UnconstrainedFESpace,c=2) = get_degree(Gridap.FESpaces.get_order(test),c)

function get_cell_quadrature(test::UnconstrainedFESpace)
  CellQuadrature(get_triangulation(test),get_degree(test))
end

struct LagrangianQuadFESpace
  test::UnconstrainedFESpace
  function LagrangianQuadFESpace(model::DiscreteModel,order::Int)
    reffe_quad = Gridap.ReferenceFE(lagrangian_quad,Float,order)
    test = TestFESpace(model,reffe_quad,conformity=:L2)
    new(test)
  end
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

struct Nonaffine <: OperatorType end
abstract type ParamVarOperator{OT,TT} end

struct ParamLinOperator{OT} <: ParamVarOperator{OT,nothing}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  tests::MyTests
end

struct ParamBilinOperator{OT,TT} <: ParamVarOperator{OT,TT}
  id::Symbol
  a::Function
  afe::Function
  pspace::ParamSpace
  trials::MyTrials{TT}
  tests::MyTests
end

get_id(op::ParamVarOperator) = op.id
get_param_function(op::ParamVarOperator) = op.a
get_fe_function(op::ParamVarOperator) = op.afe
Gridap.ODEs.TransientFETools.get_test(op::ParamVarOperator) = op.tests.test
Gridap.ODEs.TransientFETools.get_trial(op::ParamBilinOperator) = op.trials.trial
get_tests(op::ParamVarOperator) = op.tests
get_trials(op::ParamBilinOperator) = op.trials
get_test_no_bc(op::ParamVarOperator) = op.tests.test_no_bc
get_trial_no_bc(op::ParamBilinOperator) = op.trials.trial_no_bc
get_pspace(op::ParamVarOperator) = op.pspace

function Gridap.FESpaces.get_cell_dof_ids(
  op::ParamVarOperator,
  trian::Triangulation)
  collect(get_cell_dof_ids(get_test(op),trian))
end

function AffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamLinOperator{Affine}(id,a,afe,pspace,tests)
end

function NonaffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamLinOperator{Nonaffine}(id,a,afe,pspace,tests)
end

function ParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests;id=:F)
  ParamLinOperator{Nonlinear}(id,a,afe,pspace,tests)
end

function AffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{TT},tests::MyTests;id=:A) where TT
  ParamBilinOperator{Affine,TT}(id,a,afe,pspace,trials,tests)
end

function NonaffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{TT},tests::MyTests;id=:A) where TT
  ParamBilinOperator{Nonaffine,TT}(id,a,afe,pspace,trials,tests)
end

function ParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{TT},tests::MyTests;id=:A) where TT
  ParamBilinOperator{Nonlinear,TT}(id,a,afe,pspace,trials,tests)
end

function Gridap.FESpaces.assemble_vector(op::ParamLinOperator)
  afe = get_fe_function(op)
  test = get_test(op)
  μ -> assemble_vector(afe(μ),test)
end

function Gridap.FESpaces.assemble_vector(op::ParamLinOperator,t::Real)
  afe = get_fe_function(op)
  test = get_test(op)
  μ -> assemble_vector(afe(μ,t),test)
end

function Gridap.FESpaces.assemble_vector(op::ParamLinOperator,t::Vector{<:Real})
  afe = get_fe_function(op)
  test = get_test(op)
  V(μ,ti) = assemble_vector(afe(μ,ti),test)
  μ -> Matrix([V(μ,ti) for ti = t])
end

function Gridap.FESpaces.assemble_matrix(op::ParamBilinOperator)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ -> assemble_matrix(afe(μ),trial(μ),test)
end

function Gridap.FESpaces.assemble_matrix(op::ParamBilinOperator)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ -> assemble_matrix(afe(μ),trial(μ),test)
end

function Gridap.FESpaces.assemble_matrix(op::ParamBilinOperator,t::Real)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ -> assemble_matrix(afe(μ,t),trial(μ,t),test)
end

function Gridap.FESpaces.assemble_matrix(op::ParamBilinOperator,t::Vector{<:Real})
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  M(μ,ti) = assemble_matrix(afe(μ,ti),trial(μ,ti),test)
  μ -> [M(μ,ti) for ti = t]
end

realization(op::ParamVarOperator) = realization(get_pspace(op))
Gridap.Algebra.allocate_vector(op::ParamLinOperator,args...) =
  assemble_vector(op,args...)(realization(op))
Gridap.Algebra.allocate_matrix(op::ParamBilinOperator,args...) =
  assemble_matrix(op,args...)(realization(op))
allocate_structure(op::ParamLinOperator) = allocate_vector(op)
allocate_structure(op::ParamBilinOperator) = allocate_matrix(op)

function assemble_matrix_and_lifting(op::ParamBilinOperator)
  afe = get_fe_function(op)
  trial = get_trial(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ),trial_no_bc,test_no_bc)
  A_bc(μ) = A_no_bc(μ)[fdofs_test,fdofs_trial]
  dir(μ) = trial(μ).dirichlet_values
  lift(μ) = A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)

  A_bc,lift
end

function assemble_matrix_and_lifting(op::ParamBilinOperator,t::Real)
  afe = get_fe_function(op)
  trial = get_trial(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ,t),trial_no_bc,test_no_bc)
  A_bc(μ) = A_no_bc(μ)[fdofs_test,fdofs_trial]
  dir(μ) = trial(μ,t).dirichlet_values
  lift(μ) = A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)

  A_bc,lift
end

function assemble_matrix_and_lifting(op::ParamBilinOperator,t::Vector{<:Real})
  afe = get_fe_function(op)
  trial = get_trial(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ,t) = assemble_matrix(afe(μ,t),trial_no_bc,test_no_bc)
  A_bc(μ,t) = A_no_bc(μ,t)[fdofs_test,fdofs_trial]
  A_bc(μ) = [A_bc(μ,ti) for ti = t]
  dir(μ,t) = trial(μ,t).dirichlet_values
  lift(μ,t) = A_no_bc(μ,t)[fdofs_test,ddofs]*dir(μ,t)
  lift(μ) = Matrix([lift(μ,ti) for ti = t])

  A_bc,lift
end

function assemble_matrix_and_lifting(
  op::ParamBilinOperator{OT,<:UnconstrainedFESpace},args...) where OT
  assemble_matrix(op,args...)
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

Snapshots(s::Snapshots,idx) = Snapshots(s.id,getindex(s.snap,idx))

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
POD(s::Snapshots,args...;kwargs...) = POD(s.snap,args...;kwargs...)
POD(s::Vector{Snapshots},args...;kwargs...) = Broadcasting(si->POD(si,args...;kwargs...))(s)
