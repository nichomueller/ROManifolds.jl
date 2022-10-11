abstract type FEMProblem{D} end
abstract type FEMProblemS{D} <: FEMProblem{D} end
abstract type FEMProblemST{D} <: FEMProblem{D} end

abstract type Info end
abstract type ParamInfo end
abstract type ParamFormInfo end

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

struct FEMInfoS <: Info
  problem_id::NTuple
  D::Int
  problem_unknowns::Vector{String}
  problem_structures::Vector{String}
  case::Int
  probl_nl::Vector{String}
  bnd_info::Dict
  order::Int
  solver::String
  Paths::FEMPathInfo
  nₛ::Int
end

struct FEMInfoST <: Info
  problem_id::NTuple
  D::Int
  problem_unknowns::Vector{String}
  problem_structures::Vector{String}
  case::Int
  probl_nl::Vector{String}
  bnd_info::Dict
  order::Int
  solver::String
  Paths::FEMPathInfo
  nₛ::Int
  θ::Float
  t₀::Float
  tₗ::Float
  δt::Float
end

mutable struct ParamInfoS <: ParamInfo
  var::String
  fun::Function
  θ::Vector{Vector{Float}}
end

mutable struct ParamInfoST <: ParamInfo
  var::String
  funₛ::Function
  funₜ::Function
  fun::Function
  θ::Vector{Vector{Float}}
end

mutable struct ParamFormInfoS <: ParamFormInfo
  Param::ParamInfoS
  dΩ::Measure
end

mutable struct ParamFormInfoST <: ParamFormInfo
  Param::ParamInfoST
  dΩ::Measure
end

function Base.getproperty(ParamForm::ParamFormInfoS, sym::Symbol)
  if sym in (:var, :fun, :θ)
    getfield(ParamForm.Param, sym)
  else
    getfield(RBInfo, sym)
  end
end

function Base.setproperty!(ParamForm::ParamFormInfoS, sym::Symbol, x::T) where T
  if sym in (:var, :fun, :θ)
    setfield!(ParamForm.Param, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end

function Base.getproperty(ParamForm::ParamFormInfoST, sym::Symbol)
  if sym in (:var, :funₛ, :funₜ, :fun, :θ)
    getfield(ParamForm.Param, sym)
  else
    getfield(RBInfo, sym)
  end
end

function Base.setproperty!(ParamForm::ParamFormInfoST, sym::Symbol, x::T) where T
  if sym in (:var, :funₛ, :funₜ, :fun, :θ)
    setfield!(ParamForm.Param, sym, x)::T
  else
    setfield!(RBVars, sym, x)::T
  end
end
