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

struct SearchSVDRank <: ReductionStyle
  tol::Float64
end

struct FixedSVDRank <: ReductionStyle
  rank::Int
end

struct SearchTTSVDRanks <: ReductionStyle
  tols::Vector{Float64}
end

SearchTTSVDRanks(tol::Float64,N=3) = SearchTTSVDRanks(fill(tol,N))

Base.size(r::SearchTTSVDRanks) = (length(r.tols),)
Base.getindex(r::SearchTTSVDRanks,i::Integer) = SearchSVDRank(r.tols[i])

struct FixedTTSVDRanks <: ReductionStyle
  ranks::Vector{Int}
end

FixedTTSVDRanks(rank::Float64,N=3) = FixedTTSVDRanks(fill(rank,N))

Base.size(r::FixedTTSVDRanks) = (length(r.ranks),)
Base.getindex(r::FixedTTSVDRanks,i::Integer) = FixedSVDRank(r.ranks[i])

abstract type NormStyle end

get_norm(n::NormStyle) = @abstractmethod

struct EuclideanNorm <: NormStyle end

struct EnergyNorm <: NormStyle
  norm_op::Function
end

get_norm(n::EnergyNorm) = n.norm_op

abstract type AbstractReduction{A<:ReductionStyle,B<:NormStyle} end
abstract type DirectReduction{A,B} <: AbstractReduction{A,B} end
abstract type GreedyReduction{A,B} <: AbstractReduction{A,B} end

get_reduction(r::AbstractReduction) = r
ReductionStyle(r::AbstractReduction) = @abstractmethod
NormStyle(r::AbstractReduction) = @abstractmethod
ParamDataStructures.num_params(r::AbstractReduction) = @abstractmethod
get_norm(r::AbstractReduction) = get_norm(NormStyle(r))

abstract type AbstractPODReduction{A,B} <: DirectReduction{A,B} end

struct PODReduction{A,B} <: AbstractPODReduction{A,B}
  red_style::A
  norm_style::B
  nparams::Int
end

function PODReduction(red_style::ReductionStyle,norm_style::NormStyle=EuclideanNorm();nparams=50)
  PODReduction(red_style,norm_style,nparams)
end

function PODReduction(red_style::ReductionStyle,norm_op::Function;kwargs...)
  norm_style = EnergyNorm(norm_op)
  PODReduction(red_style,norm_style;kwargs...)
end

function PODReduction(tol::Float64,args...;kwargs...)
  red_style = SearchSVDRank(tol)
  PODReduction(red_style,args...;kwargs...)
end

function PODReduction(rank::Int,args...;kwargs...)
  red_style = FixedSVDRank(rank)
  PODReduction(red_style,args...;kwargs...)
end

ReductionStyle(r::PODReduction) = r.red_style
NormStyle(r::PODReduction) = r.norm_style
ParamDataStructures.num_params(r::PODReduction) = r.nparams

struct TTSVDReduction{A<:ReductionStyle,B} <: DirectReduction{A,B}
  red_style::A
  norm_style::B
  nparams::Int
end

function TTSVDReduction(red_style::ReductionStyle,norm_style::NormStyle=EuclideanNorm();nparams=50)
  TTSVDReduction(red_style,norm_style,nparams)
end

function TTSVDReduction(red_style::ReductionStyle,norm_op::Function;kwargs...)
  norm_style = EnergyNorm(norm_op)
  TTSVDReduction(red_style,norm_style;kwargs...)
end

function TTSVDReduction(tols::Vector{Float64},args...;kwargs...)
  red_style = SearchTTSVDRanks(tols)
  TTSVDReduction(red_style,args...;kwargs...)
end

function TTSVDReduction(ranks::Vector{Int},args...;D=3,kwargs...)
  red_style = FixedTTSVDRanks(ranks)
  TTSVDReduction(red_style,args...;kwargs...)
end

function TTSVDReduction(tol::Float64,args...;D=3,kwargs...)
  red_style = SearchTTSVDRanks(tol;D)
  TTSVDReduction(red_style,args...;kwargs...)
end

function TTSVDReduction(rank::Int,args...;D=3,kwargs...)
  red_style = FixedTTSVDRanks(rank;D)
  TTSVDReduction(red_style,args...;kwargs...)
end

Base.size(r::TTSVDReduction) = (length(r.tols),)
Base.getindex(r::TTSVDReduction,i::Integer) = FixedTTSVDRanks(r.ranks[i])

ReductionStyle(r::TTSVDReduction) = r.red_style
NormStyle(r::TTSVDReduction) = r.norm_style
ParamDataStructures.num_params(r::TTSVDReduction) = r.nparams

struct SupremizerReduction{A,R<:AbstractReduction{A,EnergyNorm}} <: AbstractReduction{A,EnergyNorm}
  reduction::R
  supr_op::Function
  supr_tol::Float64
end

for f in (:PODReduction,:TTSVDReduction)
  @eval begin
    function $f(red_style::ReductionStyle,norm_op::Function,supr_op::Function,args...;supr_tol=1e-2,kwargs...)
      reduction = $f(red_style,norm_op,args...;kwargs...)
      SupremizerReduction(reduction,supr_op,supr_tol)
    end

    function $f(norm_op::Function,supr_op::Function,args...;supr_tol=1e-2,kwargs...)
      reduction = $f(norm_op,args...;kwargs...)
      SupremizerReduction(reduction,supr_op,supr_tol)
    end
  end
end

get_supr(r::SupremizerReduction) = r.supr_op
get_supr_tol(r::SupremizerReduction) = r.supr_tol

get_reduction(r::SupremizerReduction) = get_reduction(r.reduction)
ReductionStyle(r::SupremizerReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::SupremizerReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::SupremizerReduction) = num_params(get_reduction(r))

# generic constructor

function Reduction(red_style::Union{SearchSVDRank,FixedSVDRank},args...;kwargs...)
  PODReduction(red_style,args...;kwargs...)
end

function Reduction(red_style::Union{SearchTTSVDRanks,FixedTTSVDRanks},args...;kwargs...)
  TTSVDReduction(red_style,args...;kwargs...)
end

abstract type AbstractMDEIMReduction{A} <: AbstractReduction{A,EuclideanNorm} end

struct MDEIMReduction{A,R<:AbstractReduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
  reduction::R
  nparams_test::Int
end

function MDEIMReduction(args...;nparams_test=10,kwargs...)
  reduction = Reduction(args...;kwargs...)
  MDEIMReduction(reduction,nparams_test)
end

get_reduction(r::MDEIMReduction) = get_reduction(r.reduction)
ReductionStyle(r::MDEIMReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::MDEIMReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::MDEIMReduction) = num_params(get_reduction(r))
num_online_params(r::MDEIMReduction) = r.nparams_test

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
- nparams_state: number of snapshots considered when running TPOD or TT-SVD
- nparams_res: number of snapshots considered when running MDEIM for the residual
- nparams_jac: number of snapshots considered when running MDEIM for the jacobian
- nparams_test:  number of snapshots considered when computing the error the RB
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
    ) where {A,B,C,D}

    @check num_online_params(residual_reduction) == num_online_params(jacobian_reduction)
    new{A,B,C,D}(fesolver,state_reduction,residual_reduction,jacobian_reduction)
  end

  function RBSolver(
    fesolver::A,
    state_reduction::B,
    residual_reduction::C,
    jacobian_reduction::D
    ) where {A,B,C,D<:Tuple}

    nparams = num_online_params(first(jacobian_reduction))
    @check all(num_online_params.(jacobian_reduction) .== nparams)
    @check num_online_params(residual_reduction) == nparams
    new{A,B,C,D}(fesolver,state_reduction,residual_reduction,jacobian_reduction)
  end
end

function RBSolver(
  fesolver::FESolver,
  state_reduction::AbstractReduction;
  nparams_res=20,
  nparams_jac=20,
  nparams_test=10)

  red_style = ReductionStyle(state_reduction)
  residual_reduction = MDEIMReduction(red_style;nparams=nparams_res,nparams_test=nparams_test)
  jacobian_reduction = MDEIMReduction(red_style;nparams=nparams_jac,nparams_test=nparams_test)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function RBSolver(fesolver::FESolver,args...;nparams_state=50,kwargs...)
  state_reduction = PODReduction(args...;nparams=nparams_state)
  RBSolver(fesolver,state_reduction,kwargs...)
end

get_fe_solver(s::RBSolver) = s.fesolver
get_state_reduction(s::RBSolver) = s.state_reduction
get_residual_reduction(s::RBSolver) = s.residual_reduction
get_jacobian_reduction(s::RBSolver) = s.jacobian_reduction

num_state_params(s::RBSolver) = num_params(s.state_reduction)
num_res_params(s::RBSolver) = num_params(s.residual_reduction)
num_jac_params(s::RBSolver) = num_params(s.jacobian_reduction)

num_offline_params(s::RBSolver) = max(num_state_params(s),num_res_params(s),num_jac_params(s))
offline_params(s::RBSolver) = 1:num_offline_params(s)
num_online_params(s::RBSolver) = num_online_params(s.residual_reduction)
online_params(s::RBSolver) = 1+num_offline_params(s):num_online_params(s)+num_offline_params(s)
res_params(s::RBSolver) = 1:num_res_params(s)
jac_params(s::RBSolver) = 1:num_jac_params(s)
ParamDataStructures.num_params(s::RBSolver) = num_offline_params(s) + num_online_params(s)

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
