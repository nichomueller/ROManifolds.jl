abstract type Problem end
abstract type FEMProblem{D} <: Problem end
abstract type FEMProblemS{D} <: FEMProblem{D} end
abstract type FEMProblemST{D} <: FEMProblem{D} end

abstract type Info end
abstract type SteadyParametricInfo <: Info end
abstract type UnsteadyParametricInfo <: Info end

struct FEMSpacePoissonS{D} <: FEMProblemS{D}
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

struct FEMSpacePoissonST{D} <: FEMProblemST{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::Function
  Nₛᵘ::Int
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpaceADRS{D} <: FEMProblemS{D}
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

struct FEMSpaceADRST{D} <: FEMProblemST{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::Function
  Nₛᵘ::Int
  Ω::BodyFittedTriangulation
  Γn::BoundaryTriangulation
  dΩ::Measure
  dΓn::Measure
  phys_quadp::Vector{Vector{VectorValue{D,Float}}}
  V₀_quad::UnconstrainedFESpace
end

struct FEMSpaceStokesS{D} <: FEMProblemS{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TrialFESpace
  Q₀::UnconstrainedFESpace
  Q::UnconstrainedFESpace
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

struct FEMSpaceStokesST{D} <: FEMProblemST{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  Q₀::UnconstrainedFESpace
  Q::UnconstrainedFESpace
  X₀::MultiFieldFESpace
  X::TransientMultiFieldTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::Function
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

struct FEMSpaceNavierStokesS{D} <: FEMProblemS{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TrialFESpace
  Q₀::UnconstrainedFESpace
  Q::UnconstrainedFESpace
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

struct FEMSpaceNavierStokesST{D} <: FEMProblemST{D}
  model::DiscreteModel
  Qₕ::CellQuadrature
  V₀::UnconstrainedFESpace
  V::TransientTrialFESpace
  Q₀::UnconstrainedFESpace
  Q::UnconstrainedFESpace
  X₀::MultiFieldFESpace
  X::TransientMultiFieldTrialFESpace
  ϕᵥ::FEBasis
  ϕᵤ::Function
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

struct InfoS <: Info
  problem_id::NTuple
  D::Int
  case::Int
  probl_nl::Vector{String}
  bnd_info::Dict
  order::Int
  solver::String
  Paths::FEMPathInfo
  nₛ::Int
end

struct InfoST <: Info
  problem_id::NTuple
  D::Int
  case::Int
  probl_nl::Vector{String}
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

struct ParamPoissonS <: SteadyParametricInfo
  μ::Vector
  α::Function
  f::Function
  g::Function
  h::Function
end

struct ParamPoissonST <: UnsteadyParametricInfo
  μ::Vector
  αₛ::Function
  αₜ::Function
  α::Function
  mₛ::Function
  mₜ::Function
  m::Function
  fₛ::Function
  fₜ::Function
  f::Function
  gₛ::Function
  gₜ::Function
  g::Function
  hₛ::Function
  hₜ::Function
  h::Function
  u₀::Function
end

struct ParamADRS <: SteadyParametricInfo
  μ::Vector
  α::Function
  b::Function
  σ::Function
  f::Function
  g::Function
  h::Function
end

struct ParamADRST <: UnsteadyParametricInfo
  μ::Vector
  αₛ::Function
  αₜ::Function
  α::Function
  mₛ::Function
  mₜ::Function
  m::Function
  bₛ::Function
  bₜ::Function
  b::Function
  σₛ::Function
  σₜ::Function
  σ::Function
  fₛ::Function
  fₜ::Function
  f::Function
  gₛ::Function
  gₜ::Function
  g::Function
  hₛ::Function
  hₜ::Function
  h::Function
  u₀::Function
end

struct ParamStokesS <: SteadyParametricInfo
  μ::Vector
  α::Function
  f::Function
  g::Function
  h::Function
end

struct ParamStokesST <: UnsteadyParametricInfo
  μ::Vector
  αₛ::Function
  αₜ::Function
  α::Function
  mₛ::Function
  mₜ::Function
  m::Function
  fₛ::Function
  fₜ::Function
  f::Function
  gₛ::Function
  gₜ::Function
  g::Function
  hₛ::Function
  hₜ::Function
  h::Function
  x₀::Function
end

struct ParamNavierStokesS <: SteadyParametricInfo
  μ::Vector
  Re::Float
  α::Function
  f::Function
  g::Function
  h::Function
end

struct ParamNavierStokesST <: UnsteadyParametricInfo
  μ::Vector
  Re::Float
  αₛ::Function
  αₜ::Function
  α::Function
  mₛ::Function
  mₜ::Function
  m::Function
  fₛ::Function
  fₜ::Function
  f::Function
  gₛ::Function
  gₜ::Function
  g::Function
  hₛ::Function
  hₜ::Function
  h::Function
  x₀::Function
end
