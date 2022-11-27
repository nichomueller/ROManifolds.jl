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
  Gμ::ParamFunctional{true})
  ParamTrialFESpace(test,Gμ.f)
end

function MyTrial(
  test::UnconstrainedFESpace,
  Gμ::ParamFunctional{false})
  ParamTransientTrialFESpace(test,Gμ.f)
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

struct Nonaffine <: OperatorType end
abstract type ParamVarOperator{OT,TT} end

struct ParamLinOperator{OT} <: ParamVarOperator{OT,nothing}
  a::Function
  afe::Function
  A::Function
  pparam::ParamSpace
  tests::MyTests
end

struct ParamBilinOperator{OT,TT} <: ParamVarOperator{OT,TT}
  a::Function
  afe::Function
  A::Vector{<:Function}
  pparam::ParamSpace
  trials::MyTrials{TT}
  tests::MyTests
end

function ParamVarOperator(
  a::Function,
  afe::Function,
  pparam::ParamSpace,
  tests::MyTests,
  OT=Nonaffine())

  A(μ) = assemble_vector(afe(μ),tests.test)
  ParamLinOperator{OT}(a,afe,A,pparam,tests)
end

function ParamVarOperator(
  a::Function,
  afe::Function,
  pparam::ParamSpace,
  trials::MyTrials{TT},
  tests::MyTests,
  OT=Nonaffine()) where TT

  A = assemble_matrix_and_lifting(afe,trials,tests)
  ParamBilinOperator{OT,TT}(a,afe,A,pparam,trials,tests)
end

function assemble_matrix_and_lifting(
  afe::Function,
  trials::MyTrials{TT},
  tests::MyTests) where TT

  U,U_no_bc,V_no_bc = trials.trial,trials.trial_no_bc,tests.test_no_bc
  fdofs,ddofs = get_fd_dofs(tests,trials)
  fdofs_test,fdofs_trial = fdofs

  A_no_bc(μ) = assemble_matrix(afe(μ),U_no_bc,V_no_bc)
  A_bc(μ) = A_no_bc(μ)[fdofs_test,fdofs_trial]
  dir(μ) = U(μ).dirichlet_values

  [μ -> A_bc(μ),μ -> A_no_bc(μ)[fdofs_test,ddofs]*dir(μ)]
end

function assemble_matrix_and_lifting(
  afe::Function,
  trials::MyTrials{TrialFESpace},
  tests::MyTests)

  [μ -> assemble_matrix(afe(μ),trials.trial,tests.test)]
end

struct Snapshot{T}
  id::Symbol
  snap::AbstractArray{T}
end

correct_path(s::Snapshot,path::String) = joinpath(path,"$(s.id).csv")
save(s::Snapshot,path::String) = writedlm(correct_path(s,path),s.snap, ','; header=false)
save(s::Snapshot,::Nothing) = @warn "Could not save variable $(s.id): no path provided"
load(path::String) = readdlm(path, ',')
function load!(s::Snapshot,path::String)
  s.snap = readdlm(correct_path(s,path), ',')
  s
end


abstract type Problem{I,S} end

struct SteadyProblem{I} <: Problem{I,true}
  μ::Snapshot{Vector{Float}}
  xh::Snapshot{Float}
  param_op::Vector{ParamVarOperator}
end

struct TimeInfo
  t0::Real
  tF::Real
  dt::Real
  θ::Real
end

get_dt(ti::TimeInfo) = ti.dt
get_θ(ti::TimeInfo) = ti.θ
get_timesθ(ti::TimeInfo) = collect(ti.t0:ti.dt:ti.tF-ti.dt).+ti.dt*ti.θ

struct UnsteadyProblem{I} <: Problem{I,false}
  μ::Snapshot{Vector{Float}}
  xh::Snapshot{Float}
  param_op::Vector{ParamVarOperator}
  time_info::TimeInfo
end

function Problem(
  μ::Snapshot,
  xh::Snapshot,
  param_op::Vector{ParamVarOperator},
  I=true)

  SteadyProblem{I}(μ,xh,param_op)
end

function Problem(
  μ::Snapshot,
  xh::Snapshot,
  param_op::Vector{ParamVarOperator},
  time_info::TimeInfo,
  I=true)

  UnsteadyProblem{I}(μ,xh,param_op,time_info)
end

function Problem(
  μ::Snapshot,
  xh::Snapshot,
  param_op::Vector{ParamVarOperator},
  t0,tF,dt,θ,
  I=true)

  time_info = TimeInfo(t0,tF,dt,θ)
  UnsteadyProblem{I}(μ,xh,param_op,time_info)
end

num_of_snapshots(p::Problem) = length(p.μ)
