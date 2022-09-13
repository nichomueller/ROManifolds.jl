abstract type Problem end
abstract type FEMProblem{D} <: Problem end
abstract type SteadyProblem{D} <: FEMProblem{D} end
abstract type UnsteadyProblem{D} <: FEMProblem{D} end

abstract type Info end
abstract type SteadyParametricInfo <: Info end
abstract type UnsteadyParametricInfo <: Info end

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
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpaceADRSteady{D} <: SteadyProblem{D}
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
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpaceADRUnsteady{D} <: UnsteadyProblem{D}
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
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpaceNavierStokesSteady{D} <: SteadyProblem{D}
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
  bnd_info::Dict
  order::Int
  solver::String
  Paths::FEMPathInfo
  nₛ::Int
end

struct UnsteadyInfo <: Info
  problem_id::NTuple
  D::Int
  case::Int
  probl_nl::Dict
  bnd_info::Dict
  order::Int
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

struct ParamPoissonSteady <: SteadyParametricInfo
  μ::Vector
  α::F
  f::F
  g::F
  h::F
end

struct ParamPoissonUnsteady <: UnsteadyParametricInfo
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

struct ParamADRSteady <: SteadyParametricInfo
  μ::Vector
  α::F
  b::F
  σ::F
  f::F
  g::F
  h::F
end

struct ParamADRUnsteady <: UnsteadyParametricInfo
  μ::Vector
  αₛ::F
  αₜ::F
  α::F
  mₛ::F
  mₜ::F
  m::F
  bₛ::F
  bₜ::F
  b::F
  σₛ::F
  σₜ::F
  σ::F
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

struct ParamStokesSteady <: SteadyParametricInfo
  μ::Vector
  α::F
  f::F
  g::F
  h::F
end

struct ParamStokesUnsteady <: UnsteadyParametricInfo
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
  x₀::F
end

struct ParamNavierStokesSteady <: SteadyParametricInfo
  μ::Vector
  Re::Float
  α::F
  f::F
  g::F
  h::F
end

struct ParamNavierStokesUnsteady <: UnsteadyParametricInfo
  μ::Vector
  Re::Float
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
  x₀::F
end
