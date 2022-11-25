abstract type ParamSolution <: GridapType end

mutable struct GenericParamSolution{C,T} <: ParamSolution
  solver::FESolver
  op::ParamOperator{C}
  uh::T
  μ::Vector{Float}
end

function Gridap.ODEs.ODETools.solve(
  solver::FESolver,
  op::ParamOperator,
  uh,
  μ::Vector{Vector{Float}})

  Broadcasting(p->solve(solver,op,uh,p))(μ)
end

function Gridap.ODEs.ODETools.solve(
  solver::FESolver,
  op::ParamOperator{C},
  uh::T,
  μ::Vector{Float}) where {C,T}

  trial = get_trial(op.feop)
  sol = GenericParamSolution{C,T}(solver,op,uh,μ)
  solve!(sol)

  ParamFESolution(sol,trial)
end

function solve!(sol::GenericParamSolution{C,T}) where {C,T}
  solver,op,uh,μ = sol.solver,sol.op,sol.uh,sol.μ
  cache = nothing
  nlop = ParamNonlinearOperator(op,uh,μ,cache)
  uh_vec = get_free_dof_values(uh)
  Gridap.Algebra.solve!(uh_vec,solver.nls,nlop,cache)
  sol
end

function solve!(sol::GenericParamSolution{Affine,T}) where T
  solver,op,uh,μ = sol.solver,sol.op,sol.uh,sol.μ
  A,b = _allocate_matrix_and_vector(op,uh)
  A = _matrix!(A,op,uh,μ)
  b = _vector!(b,op,uh,μ)
  afop = Gridap.Algebra.AffineOperator(A,b)
  cache = nothing
  newmatrix = true
  uh_vec = get_free_dof_values(uh)
  Gridap.Algebra.solve!(uh_vec,solver.ls,afop,cache,newmatrix)
  sol
end

struct ParamFESolution
  param_sol::ParamSolution
  trial
end

function Gridap.ODEs.ODETools.solve(
  solver::FESolver,
  op::ParamFEOperator,
  n=100)

  μ = realization(op,n)
  trial = get_trial(op)
  uh = zero(trial(first(μ),0.))
  param_op = get_algebraic_operator(op)

  solve(solver,param_op,uh,μ)
end
