abstract type Problem end
abstract type FEMProblem{D} <: Problem end
abstract type SteadyProblem{D} <: FEMProblem{D} end
abstract type UnsteadyProblem{D} <: FEMProblem{D} end

abstract type Info end
abstract type ParametricInfo <: Info end

const F = Function

struct FEMSpacePoissonSteady{D} <: SteadyProblem{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::FEBasis
  Nₛᵘ::Int64
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float64}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpacePoissonUnsteady{D} <: UnsteadyProblem{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::F
  Nₛᵘ::Int64
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float64}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpaceStokesSteady{D} <: SteadyProblem{D}
  model::DiscreteModel
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
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float64}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpaceStokesUnsteady{D} <: UnsteadyProblem{D}
  model::DiscreteModel
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
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float64}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpaceNavierStokesUnsteady{D} <: UnsteadyProblem{D}
  model::DiscreteModel
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
  Γn::BoundaryTriangulation
  dΩ::Measure
  Γd::BoundaryTriangulation
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float64}}}
  V₀_quad::UnconstrainedFESpace
end

struct SteadyInfo <: Info
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
  nₛ::Int64
end

struct UnsteadyInfo <: Info
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
  nₛ::Int64
  time_method::String
  θ::Float64
  RK_type::Symbol
  t₀::Float64
  tₗ::Float64
  δt::Float64
end

struct ParametricInfoSteady <: ParametricInfo
  μ::Vector
  α::F
  f::F
  g::F
  h::F
end

struct ParametricInfoUnsteady <: ParametricInfo
  μ::Vector
  αₛ::F
  αₜ::F
  α::F
  mₛ::F
  mₜ::F
  m::F
  fₛ::F
  fₜ::F
  f::F
  gₛ::F
  gₜ::F
  g::F
  hₛ::F
  hₜ::F
  h::F
  u₀::F
end
