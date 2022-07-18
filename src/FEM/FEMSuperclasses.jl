abstract type Problem end
abstract type FEMProblem{D} <: Problem end
abstract type SteadyProblem{D} <: FEMProblem{D} end
abstract type UnsteadyProblem{D} <: FEMProblem{D} end

abstract type Info end

struct FEMSpacePoissonSteady{D} <: SteadyProblem{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::FEBasis
  Nₛᵘ::Int
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpacePoissonUnsteady{D} <: UnsteadyProblem{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::F
  Nₛᵘ::Int
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
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
  Nₛᵘ::Int
  Nₛᵖ::Int
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
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
  Nₛᵘ::Int
  Nₛᵖ::Int
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
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
  Nₛᵘ::Int
  Nₛᵖ::Int
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  Γd::BoundaryTriangulation
  dΓd::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMPathInfo <: Info
  mesh_path::String
  current_test::String
  FEM_snap_path::String
  FEM_structures_path::String
end

struct SteadyInfo <: Info
  problem_id::NTuple
  D::Int
  case::Int
  probl_nl::Dict
  order::Int
  dirichlet_tags::Vector{String}
  dirichlet_bnds::Vector{Int}
  neumann_tags::Vector{String}
  neumann_bnds::Vector{Int}
  solver::String
  Paths::FEMPathInfo
  nₛ::Int
end

struct UnsteadyInfo <: Info
  problem_id::NTuple
  D::Int
  case::Int
  probl_nl::Dict
  order::Int
  dirichlet_tags::Vector{String}
  dirichlet_bnds::Vector{Int}
  neumann_tags::Vector{String}
  neumann_bnds::Vector{Int}
  solver::String
  Paths::FEMPathInfo
  nₛ::Int
  time_method::String
  θ::Float
  RK_type::Symbol
  t₀::Float
  tₗ::Float
  δt::Float
end

struct ParametricInfoSteady <: Info
  μ::Vector
  α::F
  f::F
  g::F
  h::F
end

struct ParametricInfoUnsteady <: Info
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
