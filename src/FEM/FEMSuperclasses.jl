abstract type Problem end
abstract type FEMProblem{N,T} <: Problem end
abstract type SteadyProblem{N,T} <: FEMProblem{N,T} end
abstract type UnsteadyProblem{N,T} <: FEMProblem{N,T} end

abstract type Info end
abstract type SteadyInfo{N,T} <: Info end
abstract type UnsteadyInfo{N,T} <: Info end

#= struct PoissonProblem <: FEMProblem end
struct PoissonProblemUnsteady <: FEMProblem end =#

struct FEMSpacePoissonSteady{N,T} <: SteadyProblem{N,T}
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::FEBasis
  Nₛᵘ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
end

struct FEMSpacePoissonUnsteady{N,T} <: UnsteadyProblem{N,T}
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::Function
  Nₛᵘ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
end

struct FEMSpaceStokesSteady{N,T} <: SteadyProblem{N,T}
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TrialFESpace
  Q₀::ZeroMeanFESpace
  Q::ZeroMeanFESpace
  X₀::MultiFieldFESpace
  X::MultiFieldFESpace
  ϕᵥ::FEBasis
  ϕᵤ::FEBasis
  ψᵧ::FEBasis
  ψₚ::FEBasis
  Nₛᵘ::Int64
  Nₛᵖ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  Γd::BoundaryTriangulation
  dΓd::Measure
  dΓn::Measure
end

struct FEMSpaceStokesUnsteady{N,T} <: UnsteadyProblem{N,T}
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
  dΓd::Measure
  dΓn::Measure
end

struct FEMSpaceNavierStokesUnsteady{N,T} <: UnsteadyProblem{N,T}
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
  dΓd::Measure
  dΓn::Measure
end

struct ProblemInfoSteady{N,T} <: SteadyInfo{N,T}
  problem_id::NTuple
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

struct ProblemInfoUnsteady{N,T} <: UnsteadyInfo{N,T}
  problem_id::NTuple
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
  tₗ::Float64
  δt::Float64
end

mutable struct ParametricInfoSteady{T}
  μ::Vector{T}
  model::DiscreteModel
  α::Function
  f::Function
  g::Function
  h::Function
end

mutable struct ParametricInfoUnsteady{T}
  μ::Vector{T}
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
