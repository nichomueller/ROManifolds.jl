abstract type ParamSolution <: GridapType end

mutable struct GenericParamSolution <: ParamSolution
  uh::AbstractArray
  μ::AbstractVector
  k::Int
end

function Gridap.solve!(
  solver::FESolver,
  op::ParamOp,
  uh::FEFunction,
  μ::AbstractVector)

  cache = nothing
  nlop = ParamNonlinearOperator(op,uh,μ,cache)
  sol = get_free_dof_values(uh)
  solve!(sol,solver.nls,nlop,cache)
  sol
end

function Gridap.solve!(
  solver::FESolver,
  op::ParamOp{Affine},
  uh::FEFunction,
  μ::AbstractVector)

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

function Gridap.solve(
  solver::FESolver,
  op::ParamOp,
  μk::AbstractVector,
  k::Int)

  trial = get_trial(op.feop)
  uh = zero(trial(μk))
  solk = solve!(op,solver,uh,μk)
  GenericParamSolution(solk,μk,k)
end

function Gridap.solve(
  solver::FESolver,
  op::ParamOp,
  params::Table)

  [solve(op,solver,μk,k) for (μk,k) in enumerate(params)]
end

function Gridap.solve(
  solver::FESolver,
  op::ParamFEOperator,
  n_snaps::Int)

  μ = realization(op,n_snaps)
  param_op = get_algebraic_operator(op)
  solve(solver,param_op,μ)
end
