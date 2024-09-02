"""
    create_dir(dir::String) -> Nothing

Recursive creation of a directory `dir`

"""
function create_dir(dir::String)
  if !isdir(dir)
    parent_dir, = splitdir(dir)
    create_dir(parent_dir)
    mkdir(dir)
  end
  return
end

struct TimerInfo
  name::String
  timer::TimerOutput
end

function TimerInfo(name)
  timer = TimerOutput()
  TimerInfo(name)
end

TimerOutputs.get_timer(t::TimerInfo) = t.timer
TimerOutputs.reset_timer!(t::TimerInfo) = reset_timer!(t.timer[t.name])

function set_nruns!(t::TimerInfo,nruns::Int)
  t.timer.accumulated_data.ncalls = nruns
end

abstract type AbstractReduction end
abstract type DirectReduction <: AbstractReduction end
abstract type GreedyReduction <: AbstractReduction end

TimerOutputs.get_timer(r::AbstractReduction) = @abstractmethod
TimerOutputs.reset_timer!(r::AbstractReduction) = reset_timer!(get_timer(r))
set_nruns!(r::AbstractReduction,nruns::Int) = set_nruns!(get_timer(r),nruns)
get_name(r::AbstractReduction) = get_name(get_timer(r))

get_reduction(r::AbstractReduction) = r
get_style(r::AbstractReduction) = @abstractmethod
num_snaps(r::AbstractReduction) = @abstractmethod

abstract type ReductionStyle end

struct SearchSVDRank{A} <: ReductionStyle
  tol::A
end

struct FixedSVDRank{A} <: ReductionStyle
  rank::A
end

struct PODReduction{A<:ReductionStyle} <: DirectReduction
  style::A
  nsnaps::Int
  timer::TimerInfo
end

function PODReduction(style::ReductionStyle;nsnaps=50)
  timer = TimerInfo("POD")
  PODReduction(style,nsnaps,timer)
end

function PODReduction(;style=SearchSVDRank(1e-4),kwargs...)
  PODReduction(style;kwargs...)
end

TimerOutputs.get_timer(r::PODReduction) = r.timer
get_style(r::PODReduction) = r.style
num_snaps(r::PODReduction) = r.nsnaps

struct TTSVDReduction{A<:ReductionStyle,B} <: DirectReduction
  style::A
  nsnaps::Int
  timer::TimerInfo
end

function TTSVDReduction(style::ReductionStyle;nsnaps=50)
  timer = TimerInfo("TTSVD")
  TTSVDReduction(style,nsnaps,timer)
end

function TTSVDReduction(;style=SearchSVDRank(1e-4),kwargs...)
  TTSVDReduction(style;kwargs...)
end

TimerOutputs.get_timer(r::TTSVDReduction) = r.timer
get_style(r::TTSVDReduction) = r.style
num_snaps(r::TTSVDReduction) = r.nsnaps

struct NormedReduction{A<:AbstractReduction} <: AbstractReduction
  reduction::A
  norm_op::Function
end

for f in (:PODReduction,:TTSVDReduction)
  @eval begin
    function $f(norm_op::Function,style::ReductionStyle;kwargs...)
      reduction = $f(style;kwargs...)
      NormedReduction(reduction,norm_op)
    end

    function $f(norm_op::Function;kwargs...)
      reduction = $f(;kwargs...)
      NormedReduction(reduction,norm_op)
    end
  end
end

get_norm(r::NormedReduction) = r.norm_op

get_reduction(r::NormedReduction) = get_reduction(r.reduction)
TimerOutputs.get_timer(r::NormedReduction) = get_timer(get_reduction(r))
get_style(r::NormedReduction) = get_style(get_reduction(r))
num_snaps(r::NormedReduction) = num_snaps(get_reduction(r))

const NormedPODReduction{A<:ReductionStyle} = NormedReduction{PODReduction{A}}
const NormedTTSVDReduction{A<:ReductionStyle} = NormedReduction{TTSVDReduction{A}}

struct SupremizerReduction{A<:AbstractReduction} <: AbstractReduction
  reduction::A
  supr_op::Function
  supr_tol::Float64
end

for f in (:PODReduction,:TTSVDReduction)
  @eval begin
    function $f(supr_op::Function,norm_op::Function,style::ReductionStyle;supr_tol=1e-2,kwargs...)
      reduction = $f(norm_op,style;kwargs...)
      SupremizerReduction(reduction,supr_op,supr_tol)
    end

    function $f(supr_op::Function,norm_op::Function;kwargs...)
      reduction = $f(norm_op;kwargs...)
      NormedReduction(reduction,supr_op,supr_tol)
    end
  end
end

const SupremizerPODReduction{A<:ReductionStyle} = SupremizerReduction{NormedPODReduction{A}}
const SupremizerTTSVDReduction{A<:ReductionStyle} = SupremizerReduction{NormedTTSVDReduction{A}}

get_supr(r::SupremizerReduction) = r.supr_op
get_norm(r::SupremizerReduction) = get_norm(get_reduction(r))

get_reduction(r::SupremizerReduction) = get_reduction(r.reduction)
TimerOutputs.get_timer(r::SupremizerReduction) = get_timer(get_reduction(r))
get_style(r::SupremizerReduction) = get_style(get_reduction(r))
num_snaps(r::SupremizerReduction) = num_snaps(get_reduction(r))

struct MDEIMReduction{A<:AbstractReduction} <: AbstractReduction
  style::A
  nsnaps::Int
  timer::TimerInfo
end

function MDEIMReduction(;style=SearchSVDRank(1e-4),nsnaps=20)
  timer = TimerInfo("MDEIM")
  MDEIMReduction(style,nsnaps,timer)
end

TimerOutputs.get_timer(r::MDEIMReduction) = r.timer
get_style(r::MDEIMReduction) = r.style
num_snaps(r::MDEIMReduction) = r.nsnaps

struct TestInfo
  nsnaps::Int
  timer::TimerInfo
end

function TestInfo(;nsnaps=10)
  timer = TimerInfo("TEST")
  TestInfo(nsnaps,timer)
end

TimerOutputs.get_timer(info::TestInfo) = info.timer
num_snaps(info::TestInfo) = info.nsnaps

"""
    struct RBSolver{S,M} end

Wrapper around a FE solver (e.g. [`FESolver`](@ref) or [`ODESolver`](@ref)) with
additional information on the reduced basis (RB) method employed to solve a given
problem dependent on a set of parameters. A RB method is a projection-based
reduced order model where

1) a suitable subspace of a FESpace is sought, of dimension n ≪ Nₕ
2) a matrix-based discrete empirical interpolation method (MDEEIM) is performed
  to approximate the manifold of the parametric residuals and jacobians
3) the EIM approximations are compressed with (Petrov-)Galerkin projections
  onto the subspace
4) for every desired choice of parameters, numerical integration is performed, and
  the resulting n × n system of equations is cheaply solved

In particular:

- ϵ: tolerance used in the projection-based truncated proper orthogonal
  decomposition (TPOD) or in the tensor train singular value decomposition (TT-SVD),
  where a basis spanning the reduced subspace is computed
- nsnaps_state: number of snapshots considered when running TPOD or TT-SVD
- nsnaps_res: number of snapshots considered when running MDEIM for the residual
- nsnaps_jac: number of snapshots considered when running MDEIM for the jacobian
- nsnaps_test:  number of snapshots considered when computing the error the RB
  method commits with respect to the FE procedure
- timer: cost timer for the algorithm

"""
struct RBSolver{S,A,B}
  fesolver::S
  state_reduction::A
  system_reduction::B
  test_info::TestInfo
end

function RBSolver(
  fesolver::FESolver;
  state_reduction=TTSVDReduction(),
  system_reduction=(MDEIMReduction(),MDEIMReduction()),
  test_info=TestInfo())

  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction,test_info)
end

get_fe_solver(s::RBSolver) = s.fesolver
get_state_reduction(s::RBSolver) = s.state_reduction
get_system_reduction(s::RBSolver) = s.system_reduction
get_residual_reduction(s::RBSolver) = first(get_system_reduction(s))
get_jacobian_reduction(s::RBSolver) = last(get_system_reduction(s))
get_test_info(s::RBSolver) = s.test_info

num_offline_params(s::RBSolver) = max(num_snaps(s.state_reduction),num_snaps(s.residual_reduction),num_snaps(s.jacobian_reduction))
offline_params(s::RBSolver) = 1:num_offline_params(s)
num_online_params(s::RBSolver) = num_snaps(s.test_info)
online_params(s::RBSolver) = 1+num_offline_params(s):num_online_params(s)+num_offline_params(s)
ParamDataStructures.num_params(s::RBSolver) = num_offline_params(s) + num_online_params(s)

function get_test_directory(s::RBSolver;dir=datadir())
  keyword = get_name(s.state_reduction) * "_$(get_style(s.state_reduction))"
  test_dir = joinpath(dir,keyword)
  create_dir(test_dir)
  test_dir
end

"""
    fe_solutions(solver::RBSolver,op::ParamFEOperator;kwargs...) -> AbstractSteadySnapshots
    fe_solutions(solver::RBSolver,op::TransientParamFEOperator;kwargs...) -> AbstractTransientSnapshots

The problem is solved several times, and the solution snapshots are returned along
with the information related to the computational expense of the FE method

"""
function fe_solutions(
  solver::RBSolver,
  op::ParamFEOperator;
  nparams=num_params(solver),
  r=realization(op;nparams))

  fesolver = get_fe_solver(solver)
  timer = get_timer(solver)
  reset_timer!(timer)

  index_map = get_vector_index_map(op)
  values = solve(fesolver,op,timer;r)
  snaps = Snapshots(values,index_map,r)

  return snaps
end

function Algebra.solve(rbsolver::RBSolver,feop,args...;kwargs...)
  fesnaps = fe_solutions(rbsolver,feop,args...)
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  rbsnaps = solve(rbsolver,rbop,fesnaps)
  results = rb_results(rbsolver,rbop,fesnaps,rbsnaps)
  return results
end

"""
    nonlinear_rb_solve!(x̂,x,A,b,A_cache,b_cache,dx̂,ns,nls,op,trial) -> x̂

Newton - Raphson for a RB problem

"""
function nonlinear_rb_solve!(x̂,x,A,b,A_cache,b_cache,dx̂,ns,nls,op,trial)
  A_lin, = A_cache
  max0 = maximum(abs,b)

  for k in 1:nls.max_nliters
    rmul!(b,-1)
    solve!(dx̂,ns,b)
    x̂ .+= dx̂
    x .= recast(x̂,trial)

    b = residual!(b_cache,op,x)

    A = jacobian!(A_cache,op,x)
    numerical_setup!(ns,A)

    b .+= A_lin*x̂
    maxk = maximum(abs,b)
    println(maxk)

    maxk < 1e-6*max0 && return

    if k == nls.max_nliters
      @unreachable
    end
  end
end
