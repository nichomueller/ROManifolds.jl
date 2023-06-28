abstract type ParamSolution <: GridapType end

mutable struct GenericParamSolution <: ParamSolution
  uh::AbstractArray
  μ::AbstractVector
end

function Gridap.solve!(
  op::ParamOp,
  solver::FESolver,
  uh::FEFunction,
  μ::AbstractVector)

  cache = nothing
  nlop = ParamNonlinearOperator(op,uh,μ,cache)
  sol = get_free_dof_values(uh)
  solve!(sol,solver.nls,nlop,cache)
  sol
end

function Gridap.solve!(
  op::ParamOp{Affine},
  solver::FESolver,
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
  op::ParamOp,
  solver::FESolver,
  μ::AbstractVector)

  trial = get_trial(op.feop)
  uh = zero(trial(μ))
  solk = solve!(op,solver,uh,μ)
  GenericParamSolution(solk,μ)
end

function solution_cache(test::FESpace,::FESolver)
  space_ndofs = test.nfree
  cache = fill(1.,space_ndofs,1)
  cache
end

function solution_cache(test::MultiFieldFESpace,args...)
  map(t->solution_cache(t,args...),test.spaces)
end

function collect_solution!(
  cache,
  op::ParamFEOperator,
  solver::FESolver,
  μ::AbstractVector)

  sol = solve(op,solver,μ)
  if isa(cache,AbstractMatrix)
    copyto!(cache,sol.uh)
  elseif isa(cache,Vector{AbstractMatrix})
    map((c,sol) -> copyto!(c,sol),cache,sol.uh)
  else
    @unreachable
  end
  cache
end
