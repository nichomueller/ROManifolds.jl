abstract type ParamSolution <: GridapType end

mutable struct GenericParamSolution{C} <: ParamSolution
  solver::FESolver
  op::ParamOp{C}
  uh::FEFunction
  μ::Param
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
  μ::Vector{Param})

  [solve(solver,op,p) for p in μ]
end

function Gridap.solve(
  solver::FESolver,
  op::ParamOp{C},
  μ::Param) where C

  trial = get_trial(op.feop)
  uh = zero(trial(μ))
  sol = GenericParamSolution{C}(solver,op,uh,μ)
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
get_Ns(sol::Vector{<:ParamFESolution}) = get_Ns(first(sol))
