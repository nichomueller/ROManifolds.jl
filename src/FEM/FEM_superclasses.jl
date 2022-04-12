abstract type Problem end

abstract type FEMProblem <: Problem end

struct UnsteadyProblem <: FEMProblem  end

struct SteadyProblem <: FEMProblem end

struct PoissonProblem <: FEMProblem end

struct PoissonProblemUnsteady <: FEMProblem end

struct FESpacePoisson <: FEMProblem
  Qₕ
  V₀
  V
  ϕᵥ
  ϕᵤ
  σₖ
  Nₕ
  Ω
  dΩ
  dΓ
end

struct ProblemSpecifics <: FEMProblem
  case::Int
  order::Int
  dirichlet_tags::Array
  neumann_tags::Array
  solver::String
  paths::Function
  problem_nonlinearities::Dict
end

struct ProblemSpecificsUnsteady <: FEMProblem
  case::Int
  order::Int
  dirichlet_tags::Array
  neumann_tags::Array
  solver::String

  paths::Function
  problem_nonlinearities::Dict
end

mutable struct ParametricSpecifics
  μ
  model
  α
  f
  g
  h
end
