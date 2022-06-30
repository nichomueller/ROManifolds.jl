abstract type Problem end
abstract type FEMProblem{D,T} <: Problem end
abstract type SteadyProblem{D,T} <: FEMProblem{D,T} end
abstract type UnsteadyProblem{D,T} <: FEMProblem{D,T} end

abstract type Info{T} end

abstract type ParametricInfo{T} <: Info{T} end

const F = Function

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
  phys_quadp::LazyArray
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpacePoissonUnsteady{D,T} <: UnsteadyProblem{D,T}
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::F
  Nₛᵘ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::LazyArray
  V₀_quad::UnconstrainedFESpace
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
  phys_quadp::LazyArray
  V₀_quad::UnconstrainedFESpace
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
  ϕᵤ::F
  ψᵧ::FEBasis
  ψₚ::FEBasis
  Nₛᵘ::Int64
  Nₛᵖ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  Γd::BoundaryTriangulation
  dΓd::Measure
  dΓn::Measure
  phys_quadp::LazyArray
  V₀_quad::UnconstrainedFESpace
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
  ϕᵤ::F
  ψᵧ::FEBasis
  ψₚ::FEBasis
  Nₛᵘ::Int64
  Nₛᵖ::Int64
  Ω::BodyFittedTriangulation
  dΩ::Measure
  Γd::BoundaryTriangulation
  dΓd::Measure
  dΓn::Measure
  phys_quadp::LazyArray
  V₀_quad::UnconstrainedFESpace
end

struct SteadyInfo{T} <: Info{T}
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
  paths::F
end

struct UnsteadyInfo{T} <: Info{T}
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
  paths::F
  time_method::String
  θ::Float64
  RK_type::Symbol
  t₀::Float64
  tₗ::Float64
  δt::Float64
end

struct ParametricInfoSteady{T} <: ParametricInfo{T}
  μ::Vector{T}
  model::DiscreteModel
  α::F
  f::F
  g::F
  h::F
end

struct ParametricInfoUnsteady{T} <: ParametricInfo{T}
  μ::Vector{T}
  model::DiscreteModel
  αₛ::F
  αₜ::F
  α::F
  mₛ::F
  mₜ::F
  m::F
  fₛ::F
  fₜ::F
  f::F
  g::F
  hₛ::F
  hₜ::F
  h::F
  u₀::F
end
