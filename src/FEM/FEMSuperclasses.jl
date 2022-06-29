abstract type Problem end
abstract type FEMProblem{D,T} <: Problem end
abstract type SteadyProblem{D,T} <: FEMProblem{D,T} end
abstract type UnsteadyProblem{D,T} <: FEMProblem{D,T} end

abstract type Info end

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
  phys_quadp
  V₀_quad
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
  phys_quadp
  V₀_quad
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
  phys_quadp
  V₀_quad
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
  phys_quadp
  V₀_quad
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
  phys_quadp
  V₀_quad
end

struct SteadyInfo{T} <: Info
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

struct UnsteadyInfo{T} <: Info
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
  function ParametricInfoSteady(::Type{T}) where T
    D = num_cell_dims(model)
    new{D,T}(μ, model, α, m, f, g, h)
  end
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
  function ParametricInfoUnsteady(::Type{T}) where T
    D = num_cell_dims(model)
    new{D,T}(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, g, hₛ, hₜ, h, u₀)
  end
end
