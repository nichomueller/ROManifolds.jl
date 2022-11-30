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

get_Ns(s) = num_free_dofs(s)
get_Ns(s::MultiFieldFESpace) = num_free_dofs.(s.spaces)

struct Nonaffine <: OperatorType end
abstract type ParamVarOperator{OT,TT} end

struct ParamLinOperator{OT} <: ParamVarOperator{OT,nothing}
  a::Function
  afe::Function
  pspace::ParamSpace
  tests::MyTests
end

struct ParamBilinOperator{OT,TT} <: ParamVarOperator{OT,TT}
  a::Function
  afe::Function
  pspace::ParamSpace
  trials::MyTrials{TT}
  tests::MyTests
end

get_param_function(op::ParamVarOperator) = op.a
get_fe_function(op::ParamVarOperator) = op.afe
Gridap.ODEs.TransientFETools.get_test(op::ParamVarOperator) = op.tests.test
Gridap.ODEs.TransientFETools.get_trial(op::ParamBilinOperator) = op.trials.trial
get_tests(op::ParamVarOperator) = op.tests
get_trials(op::ParamBilinOperator) = op.trials
get_test_no_bc(op::ParamVarOperator) = op.tests.test_no_bc
get_trial_no_bc(op::ParamBilinOperator) = op.trial.trial_no_bc
get_pspace(op::ParamVarOperator) = op.pspace

function Gridap.FESpaces.get_cell_dof_ids(
  op::ParamVarOperator,
  trian::Triangulation)
  collect(get_cell_dof_ids(get_test(op),trian))
end

function AffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests)
  ParamLinOperator{Affine}(a,afe,pspace,tests)
end

function NonaffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests)
  ParamLinOperator{Nonaffine}(a,afe,pspace,tests)
end

function ParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,tests::MyTests)
  ParamLinOperator{Nonlinear}(a,afe,pspace,tests)
end

function AffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{TT},tests::MyTests) where TT
  ParamBilinOperator{Affine,TT}(a,afe,pspace,trials,tests)
end

function NonaffineParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{TT},tests::MyTests) where TT
  ParamBilinOperator{Nonaffine,TT}(a,afe,pspace,trials,tests)
end

function ParamVarOperator(
  a::Function,afe::Function,pspace::ParamSpace,
  trials::MyTrials{TT},tests::MyTests) where TT
  ParamBilinOperator{Nonlinear,TT}(a,afe,pspace,trials,tests)
end

function Gridap.FESpaces.assemble_vector(op::ParamLinOperator)
  afe = get_fe_function(op)
  test = get_test(op)
  μ -> assemble_vector(afe(μ),test)
end

function Gridap.FESpaces.assemble_matrix(op::ParamBilinOperator)
  afe = get_fe_function(op)
  trial = get_trial(op)
  test = get_test(op)
  μ -> assemble_matrix(afe(μ),trial,test)
end

realization(op::ParamVarOperator) = realization(get_pspace(op))
Gridap.Algebra.allocate_vector(op::ParamLinOperator) =
  assemble_vector(op)(realization(op))
Gridap.Algebra.allocate_matrix(op::ParamBilinOperator) =
  assemble_matrix(op)(realization(op))
allocate_structure(op::ParamLinOperator) = allocate_vector(op)
allocate_structure(op::ParamBilinOperator) = allocate_matrix(op)

function assemble_lifting(op::ParamBilinOperator)
  trial = get_trial(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,_ = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ),trial_no_bc,test_no_bc)
  dir(μ) = trial(μ).dirichlet_values

  μ -> A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)
end

function assemble_matrix_and_lifting(op::ParamBilinOperator)
  trial = get_trial(op)
  trial_no_bc = get_trial_no_bc(op)
  test_no_bc = get_test_no_bc(op)
  fdofs,ddofs = get_fd_dofs(get_tests(op),get_trials(op))
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ),trial_no_bc,test_no_bc)
  A_bc(μ) = A_no_bc(μ)[fdofs_test,fdofs_trial]
  dir(μ) = trial(μ).dirichlet_values

  [μ -> A_bc(μ),μ -> A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)]
end

get_nsnap(v::AbstractVector) = length(v)
get_nsnap(m::AbstractMatrix) = size(m,2)

mutable struct Snapshots{T}
  id::Symbol
  snap::AbstractArray{T}
  nsnap::Int
  function Snapshots(id::Symbol,snap::AbstractArray{T}) where T
    new{T}(id,snap,get_nsnap(snap))
  end
end

Snapshots(s::Snapshots,idx) = Snapshots(s.id,getindex(s.snap,idx))

function allocate_snapshot(id::Symbol,::Type{T}) where T
  emat = allocate_matrix(T)
  Snapshots(id,emat)
end

get_id(s::Snapshots) = s.id
get_snap(s::Snapshots) = s.snap
get_nsnap(s::Snapshots) = s.nsnap
correct_path(path::String) = joinpath(path,".csv")
correct_path(s::Snapshots,path::String) = joinpath(path,"$(s.id).csv")
save(s,path::String) = writedlm(correct_path(path),s, ','; header=false)
save(s::Snapshots,path::String) = save(s.snap,correct_path(s,path))

function load_snap!(s::Snapshots,path::String)
  snap = load(correct_path(s,path))
  s.snap = snap
  s.nsnap = get_nsnap(snap)
  s
end

function load_snap(id::Symbol,path::String)
  s = allocate_snapshot(id,T)
  load_snap!(s,path)
end

get_Nt(s::Snapshots) = get_Nt(get_snap(s),get_nsnap(s))
mode2_unfolding(s::Snapshots) = mode2_unfolding(get_snap(s),get_nsnap(s))
POD(s::Snapshots,args...) = POD(s.snap,args...)
POD(s::Vector{Snapshots},args...) = Broadcasting(si->POD(si,args...))(s)

#= abstract type Problem{PT<:ProblemType} end

struct SteadyProblem{PT} <: Problem{PT}
  μ::Snapshots{Param}
  xh::Snapshots{Float}
  param_op::Vector{ParamVarOperator}
end =#

struct TimeInfo
  t0::Real
  tF::Real
  dt::Real
  θ::Real
end

get_dt(ti::TimeInfo) = ti.dt
get_θ(ti::TimeInfo) = ti.θ
get_timesθ(ti::TimeInfo) = collect(ti.t0:ti.dt:ti.tF-ti.dt).+ti.dt*ti.θ

#= struct UnsteadyProblem{PT} <: Problem{PT}
  μ::Snapshots{Param}
  xh::Snapshots{Float}
  param_op::Vector{ParamVarOperator}
  time_info::TimeInfo
end

function Problem(
  ::PT,
  μ::Snapshots,
  xh::Snapshots,
  param_op::Vector{ParamVarOperator})

  SteadyProblem{PT}(μ,xh,param_op)
end

function Problem(
  ::PT,
  μ::Snapshots,
  xh::Snapshots,
  param_op::Vector{ParamVarOperator},
  time_info::TimeInfo)

  UnsteadyProblem{PT}(μ,xh,param_op,time_info)
end

function Problem(
  ::PT,
  μ::Snapshots,
  xh::Snapshots,
  param_op::Vector{ParamVarOperator},
  t0,tF,dt,θ)

  time_info = TimeInfo(t0,tF,dt,θ)
  UnsteadyProblem{PT}(μ,xh,param_op,time_info)
end =#
