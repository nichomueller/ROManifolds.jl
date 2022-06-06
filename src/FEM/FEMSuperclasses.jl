abstract type Problem end
abstract type FEMProblem <: Problem end
abstract type SteadyProblem <: FEMProblem end
abstract type UnsteadyProblem <: FEMProblem  end
abstract type Info end
abstract type SteadyInfo <: Info end
abstract type UnsteadyInfo <: Info end

#= struct PoissonProblem <: FEMProblem end
struct PoissonProblemUnsteady <: FEMProblem end =#

struct FESpacePoissonSteady <: SteadyProblem
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::FEBasis
  Nₛᵘ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  dΓd::Union{Measure,Nothing}
  dΓn::Union{Measure,Nothing}
end

struct FESpacePoissonUnsteady <: UnsteadyProblem
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::Function
  Nₛᵘ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  dΓd::Union{Measure,Nothing}
  dΓn::Union{Measure,Nothing}
end

struct FESpaceStokesSteady <: SteadyProblem
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TrialFESpace
  Q₀::ZeroMeanFESpace
  Q::ZeroMeanFESpace
  X₀#::MultiFieldFESpace
  X#::MultiFieldTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::FEBasis
  ψᵧ::FEBasis
  ψₚ::FEBasis
  Nₛᵘ::Int64
  Nₛᵖ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  Γd::BoundaryTriangulation
  dΓd::Union{Measure,Nothing}
  dΓn::Union{Measure,Nothing}
end

struct FESpaceStokesUnsteady <: UnsteadyProblem
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  Q₀::ZeroMeanFESpace
  Q::ZeroMeanFESpace
  X₀::MultiFieldFESpace
  X::TransientMultiFieldTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::Function
  ψᵧ::FEBasis
  ψₚ::FEBasis
  Nₛᵘ::Int64
  Nₛᵖ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  Γd::BoundaryTriangulation
  dΓd::Union{Measure,Nothing}
  dΓn::Union{Measure,Nothing}
end

struct ProblemInfoSteady <: SteadyInfo
  case::Int
  probl_nl::Dict
  order::Int
  dirichlet_tags::Array
  dirichlet_bnds::Array
  neumann_tags::Array
  neumann_bnds::Array
  solver::String
  paths::Function
end

struct ProblemInfoUnsteady <: UnsteadyInfo
  case::Int
  probl_nl::Dict
  order::Int
  dirichlet_tags::Array
  dirichlet_bnds::Array
  neumann_tags::Array
  neumann_bnds::Array
  solver::String
  paths::Function
  time_method::String
  θ::Float64
  RK_type::Symbol
  t₀::Float64
  T::Float64
  δt::Float64
end

mutable struct ParametricInfoSteady
  μ::Vector
  model::DiscreteModel
  α::Function
  f::Function
  g::Function
  h::Function
end

mutable struct ParametricInfoUnsteady
  μ::Vector
  model::DiscreteModel
  αₛ::Function
  αₜ::Function
  α::Function
  mₛ::Function
  mₜ::Function
  m::Function
  fₛ::Function
  fₜ::Function
  f::Function
  g::Function
  hₛ::Function
  hₜ::Function
  h::Function
  u₀::Function
end
