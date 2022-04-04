abstract type FEMProblem end

struct UnsteadyProblem <: FEMProblem  end

struct SteadyProblem <: FEMProblem end

struct FSIProblem <: FEMProblem end

struct NavierStokesProblem <: FEMProblem end

struct StokesProblem <: FEMProblem end

struct ADRProblem <: FEMProblem end

struct DiffusionProblem <: FEMProblem end

struct ConservationLawProblem <: FEMProblem end

struct PoissonProblem <: FEMProblem end


struct FESpacePoisson <: FEMProblem
  Qₕ
  V₀
  V
  ϕᵥ
  ϕᵤ
  σₖ
  Nₕ
  dΩ
  dΓ
end

struct problem_specifics
  order::Int
  dirichlet_tags::Array
  neumann_tags::Array
  solver::String
  paths::Function
  problem_nonlinearities::Dict
end

struct parametric_specifics
  μ::Array
  model::UnstructuredDiscreteModel
  α::Function
  f::Function
  g::Function
  h::Function
end
