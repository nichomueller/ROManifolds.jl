abstract type ParamSolution <: GridapType end

mutable struct GenericParamSolution <: ParamSolution
  uh::AbstractArray
  μ::AbstractVector
end

function solve!(
  op::ParamOperator,
  solver::FESolver,
  uh::T,
  μ::AbstractVector) where T

  cache = nothing
  nlop = ParamNonlinearOperator(op,uh,μ,cache)
  sol = get_free_dof_values(uh)
  solve!(sol,solver.nls,nlop,cache)
  sol
end

function solve!(
  op::ParamOperator{Affine},
  solver::FESolver,
  uh::T,
  μ::AbstractVector) where T

  A,b = _allocate_matrix_and_vector(op,uh)
  A = _matrix!(A,op,uh,μ)
  b = _vector!(b,op,uh,μ)
  afop = AffineOperator(A,b)
  cache = nothing
  newmatrix = true
  sol = get_free_dof_values(uh)
  solve!(sol,solver.ls,afop,cache,newmatrix)
  sol
end

function solve(
  op::ParamOperator,
  solver::FESolver,
  μ::AbstractVector)

  trial = get_trial(op.feop)
  uh = zero(trial(μ))
  solk = solve!(op,solver,uh,μ)
  GenericParamSolution(solk,μ)
end
