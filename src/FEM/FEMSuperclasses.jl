abstract type Problem end
abstract type FEMProblem{D,T} <: Problem end
abstract type SteadyProblem{D,T} <: FEMProblem{D,T} end
abstract type UnsteadyProblem{D,T} <: FEMProblem{D,T} end

abstract type Info end
abstract type SteadyInfo{T} <: Info end
abstract type UnsteadyInfo{T} <: Info end

struct FEMSpacePoissonSteady{D,T} <: SteadyProblem{D,T}
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

struct FEMSpacePoissonUnsteady{D,T} <: UnsteadyProblem{D,T}
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

struct FEMSpaceStokesSteady{D,T} <: SteadyProblem{D,T}
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

struct FEMSpaceStokesUnsteady{D,T} <: UnsteadyProblem{D,T}
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

struct FEMSpaceNavierStokesUnsteady{D,T} <: UnsteadyProblem{D,T}
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

struct ProblemInfoSteady{T} <: SteadyInfo{T}
  problem_id::NTuple
  D::Int64
  case::Int
  probl_nl::Dict
  order::Int
  dirichlet_tags::Vector{String}
  dirichlet_bnds::Vector{Int64}
  neumann_tags::Vector{String}
  neumann_bnds::Vector{Int64}
  solver::String
  paths::Function
end

struct ProblemInfoUnsteady{T} <: UnsteadyInfo{T}
  S::ProblemInfoSteady{T}
  time_method::String
  θ::Float64
  RK_type::Symbol
  t₀::Float64
  tₗ::Float64
  δt::Float64
end

mutable struct ParametricInfoSteady{D,T}
  μ::Vector{T}
  model::DiscreteModel{D,D}
  α::Function
  f::Function
  g::Function
  h::Function
end

mutable struct ParametricInfoUnsteady{D,T}
  μ::Vector{T}
  model::DiscreteModel{D,D}
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
