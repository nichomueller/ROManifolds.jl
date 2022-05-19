abstract type Problem end

abstract type Params end

abstract type FEMProblem <: Problem end

abstract type SteadyProblem <: FEMProblem end

abstract type UnsteadyProblem <: FEMProblem  end

struct PoissonProblem <: FEMProblem end

struct PoissonProblemUnsteady <: FEMProblem end

struct FESpacePoisson <: SteadyProblem
  Qₕ::CellQuadrature
  V₀::Gridap.FESpaces.UnconstrainedFESpace
  V::TrialFESpace
  ϕᵥ::Gridap.FESpaces.SingleFieldFEBasis
  ϕᵤ::Gridap.FESpaces.SingleFieldFEBasis
  Nₛᵘ::Int64
  Ω::Gridap.Geometry.BodyFittedTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
end

struct FESpacePoissonUnsteady <: UnsteadyProblem
  Qₕ::CellQuadrature
  V₀::Gridap.FESpaces.UnconstrainedFESpace
  V::TransientTrialFESpace
  ϕᵥ::Gridap.FESpaces.SingleFieldFEBasis
  ϕᵤ::Function
  σᵤ::Gridap.Arrays.Table
  Nₛᵘ::Int64
  Ω::Gridap.Geometry.BodyFittedTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
end

struct FESpaceStokes <: SteadyProblem
  Qₕ::CellQuadrature
  V₀::Gridap.FESpaces.UnconstrainedFESpace
  V::TrialFESpace
  Q₀::Gridap.FESpaces.UnconstrainedFESpace
  Q::TrialFESpace
  ϕᵥ::Gridap.FESpaces.SingleFieldFEBasis
  ϕᵤ::Gridap.FESpaces.SingleFieldFEBasis
  ψᵧ::Gridap.FESpaces.SingleFieldFEBasis
  ψₚ::Gridap.FESpaces.SingleFieldFEBasis
  σᵤ::Gridap.Arrays.Table
  Nₛᵘ::Int64
  Nₛᵖ::Int64
  Ω::Gridap.Geometry.BodyFittedTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
end

struct FESpaceStokesUnsteady <: UnsteadyProblem
  Qₕ::CellQuadrature
  V₀::Gridap.FESpaces.UnconstrainedFESpace
  V::TransientTrialFESpace
  Q₀::Gridap.FESpaces.UnconstrainedFESpace
  Q::TrialFESpace
  ϕᵥ::Gridap.FESpaces.SingleFieldFEBasis
  ϕᵤ::Function
  ψᵧ::Gridap.FESpaces.SingleFieldFEBasis
  ψₚ::Gridap.FESpaces.SingleFieldFEBasis
  σᵤ::Gridap.Arrays.Table
  Nₛᵘ::Int64
  Nₛᵖ::Int64
  Ω::Gridap.Geometry.BodyFittedTriangulation
  dΩ::Measure
  dΓd::Measure
  dΓn::Measure
end

struct ProblemSpecifics <: SteadyProblem
  case::Int
  probl_nl::Dict
  order::Int
  dirichlet_tags::Array
  dirichlet_labels::Array
  neumann_tags::Array
  neumann_labels::Array
  solver::String
  paths::Function
end

struct ProblemSpecificsUnsteady <: UnsteadyProblem
  case::Int
  probl_nl::Dict
  order::Int
  dirichlet_tags::Array
  dirichlet_labels::Array
  neumann_tags::Array
  neumann_labels::Array
  solver::String
  paths::Function
  time_method::String
  θ::Float64
  RK_type::Symbol
  t₀::Float64
  T::Float64
  δt::Float64
end

mutable struct ParametricSpecifics <: Params
  μ::Array
  model::UnstructuredDiscreteModel
  α::Function
  f::Function
  g::Function
  h::Function
end

mutable struct ParametricSpecificsUnsteady <: Params
  μ::Array
  model::UnstructuredDiscreteModel
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
