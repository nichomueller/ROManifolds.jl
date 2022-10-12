abstract type FOM{D} end
abstract type FOMS{D} <: FOM{D} end
abstract type FOMST{D} <: FOM{D} end

abstract type FOMInfo end
abstract type ParamInfo end
abstract type ParamFormInfo end

struct FOMPoissonS{D} <: FOMS{D}
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

struct FOMPoissonST{D} <: FOMST{D}
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

struct FOMStokesS{D} <: FOMS{D}
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

struct FOMStokesST{D} <: FOMST{D}
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

struct FOMNavierStokesS{D} <: FOMS{D}
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

struct FOMNavierStokesST{D} <: FOMST{D}
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

struct FOMPath
  mesh_path::String
  current_test::String
  FEM_snap_path::String
  FEM_structures_path::String
end

function FOMPath(root, steadiness, name, mesh_name, case)

  @assert isdir(root) "$root is an invalid root directory"

  root_tests = joinpath(root, "tests")
  create_dir(root_tests)
  mesh_path = joinpath(root_tests, joinpath("meshes", mesh_name))
  @assert isfile(mesh_path) "$mesh_path is an invalid mesh path"
  type_path = joinpath(root_tests, steadiness)
  create_dir(type_path)
  problem_path = joinpath(type_path, name)
  create_dir(problem_path)
  problem_and_info_path = joinpath(problem_path, "case$case")
  create_dir(problem_and_info_path)
  current_test = joinpath(problem_and_info_path, mesh_name)
  create_dir(current_test)
  FEM_path = joinpath(current_test, "FEM_data")
  create_dir(FEM_path)
  FEM_snap_path = joinpath(FEM_path, "snapshots")
  create_dir(FEM_snap_path)
  FEM_structures_path = joinpath(FEM_path, "FEM_structures")
  create_dir(FEM_structures_path)

  FEMPathInfo(mesh_path, current_test, FEM_snap_path, FEM_structures_path)

end

struct FOMInfoS <: FOMInfo
  id::NTuple
  D::Int
  unknowns::Vector{String}
  structures::Vector{String}
  affine_structures::Vector{String}
  bnd_info::Dict
  order::Int
  solver::String
  Paths::FEMPathInfo
  nₛ::Int
end

struct FOMInfoST <: FOMInfo
  id::NTuple
  D::Int
  unknowns::Vector{String}
  structures::Vector{String}
  affine_structures::Vector{String}
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

function ParamInfo(
  ::FOMInfoS,
  fun::Function,
  var::String)

  ParamInfoS(var, fun, Vector{Float}[])

end

function ParamInfo(
  ::FOMInfoST,
  fun::Function,
  var::String)

  ParamInfoST(var, fun, x->fun(x,t), t->fun(x,t), Vector{Float}[])

end

function ParamInfo(
  FEMInfo::FOMInfoS,
  μ::Vector,
  var::String)

  fun = get_fun(FEMInfo, μ, var)
  ParamInfoS(var, fun, Vector{Float}[])

end

function ParamInfo(
  FEMInfo::FOMInfoST,
  μ::Vector,
  var::String)

  funₛ, funₜ, fun  = get_fun(FEMInfo, μ, var)
  ParamInfoST(var, funₛ, funₜ, fun, Vector{Float}[])

end

function ParamInfo(
  FEMInfo::FOMInfo,
  μ::Vector,
  operators::Vector{String})

  get_single_ParamInfo(var) = ParamInfo(FEMInfo, μ, var)
  Broadcasting(get_single_ParamInfo)(operators)

end

function ParamInfo(
  Params::Vector{ParamInfo},
  var::String)

  for Param in Params
    if Param.var == var
      return Param
    end
  end

  error("Unrecognized variable")

end

mutable struct ParamFormInfoS <: ParamFormInfo
  Param::ParamInfoS
  dΩ::Measure
end

mutable struct ParamFormInfoST <: ParamFormInfo
  Param::ParamInfoST
  dΩ::Measure
end

function ParamFormInfo(
  FEMSpace::FOMS,
  Param::ParamInfoS)

  ParamFormInfoS(Param, get_measure(FEMSpace, var))

end

function ParamFormInfo(
  FEMSpace::FOMST,
  Param::ParamInfoST)

  ParamFormInfoST(Param, get_measure(FEMSpace, var))

end

function ParamFormInfo(
  dΩ::Measure,
  Param::ParamInfoS)

  ParamFormInfoS(Param, dΩ)

end

function ParamFormInfo(
  dΩ::Measure,
  Param::ParamInfoST)

  ParamFormInfoST(Param, dΩ)

end

function ParamFormInfo(
  FEMInfo::FOMInfo,
  μ::Vector)

  get_single_ParamFormInfo(var) = ParamInfo(FEMInfo, μ, var)
  operators = get_FEM_structures(FEMInfo)
  Broadcasting(get_single_ParamFormInfo)(operators)

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
