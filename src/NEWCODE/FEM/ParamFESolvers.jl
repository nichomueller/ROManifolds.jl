abstract type ParamSolution <: GridapType end

mutable struct GenericParamSolution{C} <: ParamSolution
  solver::FESolver
  op::ParamOp{C}
  uh::FEFunction
  μ::AbstractVector
  k::Int
end

struct ParamFESolution
  psol::ParamSolution
  trial
end

function Gridap.solve!(sol::GenericParamSolution{C}) where C
  solver,op,uh,μ = sol.solver,sol.op,sol.uh,sol.μ
  cache = nothing
  nlop = ParamNonlinearOperator(op,uh,μ,cache)
  uh_vec = get_free_dof_values(uh)
  solve!(uh_vec,solver.nls,nlop,cache)
  sol
end

function Gridap.solve!(sol::GenericParamSolution{Affine})
  solver,op,uh,μ = sol.solver,sol.op,sol.uh,sol.μ
  A,b = _allocate_matrix_and_vector(op,uh)
  A = _matrix!(A,op,uh,μ)
  b = _vector!(b,op,uh,μ)
  afop = AffineOperator(A,b)
  cache = nothing
  newmatrix = true
  uh_vec = get_free_dof_values(uh)
  solve!(uh_vec,solver.ls,afop,cache,newmatrix)
  sol
end

function Gridap.solve(
  solver::FESolver,
  op::ParamOp,
  params::Table{Float,Vector{Float},Vector{Int32}})

  [solve(solver,op,μk,k) for (μk,k) in enumerate(params)]
end

function Gridap.solve(
  solver::FESolver,
  op::ParamOp{C},
  μk::AbstractVector,
  k::Int) where C

  trial = get_trial(op.feop)
  uh = zero(trial(μk))
  sol = GenericParamSolution{C}(solver,op,uh,μk,k)
  solve!(sol)

  ParamFESolution(sol,trial)
end

function Gridap.solve(
  solver::FESolver,
  op::ParamFEOperator,
  n=100)

  μ = realization(op,n)
  param_op = get_algebraic_operator(op)

  solve(solver,param_op,μ)
end

get_Nt(sol::ParamFESolution) = 1
get_Ns(sol::ParamFESolution) = get_Ns(sol.psol.op.feop)
