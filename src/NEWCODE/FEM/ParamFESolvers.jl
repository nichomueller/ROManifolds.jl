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
  solk = solve!(solver,op,uh,μk)
  GenericParamSolution(solk,μk,k)
end

function Gridap.solve(
  solver::FESolver,
  op::ParamOp,
  params::Table)

  [solve(solver,op,μk,k) for (μk,k) in enumerate(params)]
end

function Gridap.solve(
  solver::FESolver,
  op::ParamFEOperator,
  n_snaps::Int)

  params = realization(op,n_snaps)
  param_op = get_algebraic_operator(op)
  solve(solver,param_op,params),params
end

function solution_cache(test::FESpace,::FESolver)
  space_ndofs = test.nfree
  cache = fill(1.,space_ndofs,1)
  NnzArray(cache)
end

function solution_cache(test::MultiFieldFESpace,args...)
  map(t->solution_cache(t,args...),test.spaces)
end

function collect_snapshot!(cache,sol::ParamSolution)
  printstyled("Computing snapshot $(sol.k)\n";color=:blue)
  if isa(cache,NnzArray)
    copyto!(cache,sol.uh)
  else
    map((c,sol) -> copyto!(c,sol),cache,sol.uh)
  end
  printstyled("Successfully computed snapshot $(sol.k)\n";color=:blue)

  cache
end
