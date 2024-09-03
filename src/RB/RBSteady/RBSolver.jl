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

abstract type ReductionStyle end

struct SearchSVDRank{A} <: ReductionStyle
  tol::A
end

struct FixedSVDRank{A} <: ReductionStyle
  rank::A
end

abstract type NormStyle end

get_norm(n::NormStyle) = @abstractmethod

struct EuclideanNorm <: NormedStyle end

struct EnergyNorm <: NormedStyle
  norm_op::Function
end

get_norm(n::EnergyNorm) = n.norm_op

abstract type AbstractReduction{A<:ReductionStyle,B<:NormStyle} end
abstract type DirectReduction{A,B} <: AbstractReduction{A,B} end
abstract type GreedyReduction{A,B} <: AbstractReduction{A,B} end

get_reduction(r::AbstractReduction) = r
ReductionStyle(r::AbstractReduction) = @abstractmethod
NormStyle(r::AbstractReduction) = @abstractmethod
num_snaps(r::AbstractReduction) = @abstractmethod
get_norm(r::AbstractReduction) = get_norm(NormStyle(r))

struct PODReduction{A,B} <: DirectReduction{A,B}
  red_style::A
  norm_style::B
  nsnaps::Int
end

function PODReduction(red_style::ReductionStyle,norm_style::NormStyle;nsnaps=50)
  PODReduction(red_style,norm_style,nsnaps)
end

function PODReduction(red_style::ReductionStyle,norm_op::Function;kwargs...)
  norm_style = EnergyNorm(norm_op)
  PODReduction(red_style,norm_style;kwargs...)
end

function PODReduction(norm_op::Function;red_style=SearchSVDRank(1e-4),kwargs...)
  norm_style = EnergyNorm(norm_op)
  PODReduction(red_style,norm_style;kwargs...)
end

function PODReduction(;red_style=SearchSVDRank(1e-4),norm_style=EuclideanNormReduction(),kwargs...)
  PODReduction(red_style,norm_style;kwargs...)
end

ReductionStyle(r::PODReduction) = r.red_style
NormStyle(r::PODReduction) = r.norm_style
num_snaps(r::PODReduction) = r.nsnaps

struct TTSVDReduction{A<:ReductionStyle,B} <: DirectReduction
  red_style::A
  nsnaps::Int
end

function TTSVDReduction(red_style::ReductionStyle;nsnaps=50,)
  TTSVDReduction(red_style,nsnaps)
end

function TTSVDReduction(red_style::ReductionStyle,norm_op::Function;kwargs...)
  norm_style = EnergyNorm(norm_op)
  TTSVDReduction(red_style,norm_style;kwargs...)
end

function TTSVDReduction(norm_op::Function;red_style=SearchSVDRank(1e-4),kwargs...)
  norm_style = EnergyNorm(norm_op)
  TTSVDReduction(red_style,norm_style;kwargs...)
end

function TTSVDReduction(;red_style=SearchSVDRank(1e-4),norm_style=EuclideanNormReduction(),kwargs...)
  TTSVDReduction(red_style,norm_style;kwargs...)
end

ReductionStyle(r::TTSVDReduction) = r.red_style
NormStyle(r::TTSVDReduction) = r.norm_style
num_snaps(r::TTSVDReduction) = r.nsnaps

struct SupremizerReduction{A,R<:AbstractReduction{A,EnergyNorm}} <: AbstractReduction{A,EnergyNorm}
  reduction::R
  supr_op::Function
  supr_tol::Float64
end

for f in (:PODReduction,:TTSVDReduction)
  @eval begin
    function $f(supr_op::Function,norm_op::Function,red_style::ReductionStyle;supr_tol=1e-2,kwargs...)
      reduction = $f(norm_op,red_style;kwargs...)
      SupremizerReduction(reduction,supr_op,supr_tol)
    end

    function $f(supr_op::Function,norm_op::Function;kwargs...)
      reduction = $f(norm_op;kwargs...)
      SupremizerReduction(reduction,supr_op,supr_tol)
    end
  end
end

get_supr(r::SupremizerReduction) = r.supr_op
get_supr_tol(r::SupremizerReduction) = r.supr_tol

get_reduction(r::SupremizerReduction) = get_reduction(r.reduction)
ReductionStyle(r::SupremizerReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::SupremizerReduction) = NormStyle(get_reduction(r))
num_snaps(r::SupremizerReduction) = num_snaps(get_reduction(r))

abstract type AbstractMDEIMReduction{A} <: AbstractReduction{A,EuclideanNorm} end

struct MDEIMReduction{A,R<:AbstractReduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
  reduction::R
  online_nsnaps::Int
end

function MDEIMReduction(;
  red_style=SearchSVDRank(1e-4),
  nsnaps=20,
  online_nsnaps=10,
  reduction=PODReduction(;red_style,nsnaps))

  MDEIMReduction(reduction,online_nsnaps)
end

get_reduction(r::MDEIMReduction) = get_reduction(r.reduction)
ReductionStyle(r::MDEIMReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::MDEIMReduction) = NormStyle(get_reduction(r))
num_snaps(r::MDEIMReduction) = num_snaps(get_reduction(r))
num_online_snaps(r::MDEIMReduction) = r.online_nsnaps

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

"""
struct RBSolver{A,B,C,D}
  fesolver::A
  state_reduction::B
  residual_reduction::C
  jacobian_reduction::D

  function RBSolver(
    fesolver::A,
    state_reduction::B,
    residual_reduction::C,
    jacobian_reduction::D
    ) where {A,B,C,D<:AbstractReduction}

    @check num_online_snaps(residual_reduction) == num_online_snaps(jacobian_reduction)
    new{A,B,C,D}(fesolver,state_reduction,residual_reduction,jacobian_reduction)
  end

  function RBSolver(
    fesolver::A,
    state_reduction::B,
    residual_reduction::C,
    jacobian_reduction::D
    ) where {A,B,C,D<:Tuple{Vararg{AbstractReduction}}}

    nsnaps = num_online_snaps(first(num_online_snaps))
    @check all(num_online_snaps.(jacobian_reduction) .== nsnaps)
    @check num_online_snaps(residual_reduction) == nsnaps
    new{A,B,C,D}(fesolver,state_reduction,residual_reduction,jacobian_reduction)
  end
end

function PODRBSolver(
  fesolver::FESolver,
  red_style::ReductionStyle=SearchSVDRank(1e-4),
  norm_style::NormStyle=EuclideanNorm();
  nsnaps_state=50,
  nsnaps_res=20,
  snaps_jac=20,
  nsnaps_test=10)

  state_reduction = PODReduction(red_style,norm_style,nsnaps_state)
  residual_reduction = MDEIMReduction(PODReduction(red_style,nsnaps_res))
  jacobian_reduction = MDEIMReduction(;red_style,nsnaps=jac_snaps)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

get_fe_solver(s::RBSolver) = s.fesolver
get_state_reduction(s::RBSolver) = s.state_reduction
get_residual_reduction(s::RBSolver) = s.residual_reduction
get_jacobian_reduction(s::RBSolver) = s.jacobian_reduction

num_offline_params(s::RBSolver) = max(num_snaps(s.state_reduction),num_snaps(s.residual_reduction),num_snaps(s.jacobian_reduction))
offline_params(s::RBSolver) = 1:num_offline_params(s)
num_online_params(s::RBSolver) = num_online_snaps(s.residual_reduction)
online_params(s::RBSolver) = 1+num_offline_params(s):num_online_params(s)+num_offline_params(s)
res_params(s::RBSolver) = 1:num_snaps(get_residual_reduction(s))
jac_params(s::RBSolver) = 1:num_snaps(get_jacobian_reduction(s))
ParamDataStructures.num_params(s::RBSolver) = num_offline_params(s) + num_online_params(s)

function get_test_directory(s::RBSolver;dir=datadir())
  keyword = get_name(s.state_reduction) * "_$(ReductionStyle(s.state_reduction))"
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
  index_map = get_vector_index_map(op)
  values,stats = solve(fesolver,op;r)
  snaps = Snapshots(values,index_map,r)

  return snaps,stats
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
