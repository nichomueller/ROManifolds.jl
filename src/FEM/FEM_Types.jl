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

  FOMPath(mesh_path, current_test, FEM_snap_path, FEM_structures_path)

end

abstract type FOMInfo{ID} end

struct FOMInfoS{ID} <: FOMInfo{ID}
  D::Int
  unknowns::Vector{String}
  structures::Vector{String}
  affine_structures::Vector{String}
  bnd_info::Dict
  order::Int
  solver::String
  Paths::FOMPath
  nₛ::Int
end

struct FOMInfoST{ID} <: FOMInfo{ID}
  D::Int
  unknowns::Vector{String}
  structures::Vector{String}
  affine_structures::Vector{String}
  bnd_info::Dict
  order::Int
  solver::String
  Paths::FOMPath
  nₛ::Int
  θ::Float
  t₀::Float
  tₗ::Float
  δt::Float
end

abstract type FOM{D} end
abstract type FOMS{D} <: FOM{D} end
abstract type FOMST{D} <: FOM{D} end

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

function FOMPoissonS(
  FEMInfo::FOMInfoS{1},
  model::DiscreteModel{D,D},
  g::Function) where D

  Ω, Γn, Qₕ, dΩ, dΓn = get_mod_meas_quad(FEMInfo, model)

  refFE = Gridap.ReferenceFE(lagrangian, Float, FEMInfo.order)
  V₀ = TestFESpace(model, refFE; conformity=:H1,
    dirichlet_tags=["dirichlet"])
  V = TrialFESpace(V₀, g)
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  FOMPoissonS{D}(
    model, Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, Γn, dΩ, dΓn, phys_quadp, V₀_quad)

end

function FOMPoissonS(
  FEMInfo::FOMInfoS{1},
  model::DiscreteModel)

  FOMPoissonS(FEMInfo, model, get_g₀(FEMInfo))

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

function FOMPoissonST(
  FEMInfo::FOMInfoST{1},
  model::DiscreteModel{D,D},
  g::Function) where D

  Ω, Γn, Qₕ, dΩ, dΓn = get_mod_meas_quad(FEMInfo, model)

  refFE = Gridap.ReferenceFE(lagrangian, Float, FEMInfo.order)
  V₀ = TestFESpace(model, refFE; conformity=:H1,
    dirichlet_tags=["dirichlet"])
  V = TransientTrialFESpace(V₀, g)
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  FOMPoissonST{D}(
    model, Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, Γn, dΩ, dΓn, phys_quadp, V₀_quad)

end

function FOMPoissonST(
  FEMInfo::FOMInfoST{1},
  model::DiscreteModel)

  FOMPoissonST(FEMInfo, model, get_g₀(FEMInfo))

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

function FOMStokesS(
  FEMInfo::FOMInfoS{2},
  model::DiscreteModel{D,D},
  g::Function) where D

  Ω, Γn, Qₕ, dΩ, dΓn = get_mod_meas_quad(FEMInfo, model)

  refFEᵤ = Gridap.ReferenceFE(lagrangian, VectorValue{D,Float}, FEMInfo.order)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1, dirichlet_tags=["dirichlet"])
  V = TrialFESpace(V₀, g)
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  refFEₚ = Gridap.ReferenceFE(lagrangian, Float, FEMInfo.order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ; conformity=:L2)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = MultiFieldFESpace([V, Q])

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  FOMStokesS{D}(model, Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ,
    Ω, Γn, dΩ, dΓn, phys_quadp, V₀_quad)

end

function FOMStokesS(
  FEMInfo::FOMInfoS{2},
  model::DiscreteModel)

  FOMStokesS(FEMInfo, model, get_g₀(FEMInfo))

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

function FOMStokesST(
  FEMInfo::FOMInfoST{2},
  model::DiscreteModel{D,D},
  g::Function) where D

  Ω, Γn, Qₕ, dΩ, dΓn = get_mod_meas_quad(FEMInfo, model)

  refFEᵤ = Gridap.ReferenceFE(lagrangian, VectorValue{D,Float}, FEMInfo.order)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1, dirichlet_tags=["dirichlet"])
  V = TransientTrialFESpace(V₀, g)
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  refFEₚ = Gridap.ReferenceFE(lagrangian, Float, FEMInfo.order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ; conformity=:L2)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q₀))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = TransientMultiFieldFESpace([V, Q])

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  FOMStokesST{D}(model, Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ,
    Ω, Γn, dΩ, dΓn, phys_quadp, V₀_quad)

end

function FOMStokesST(
  FEMInfo::FOMInfoST{2},
  model::DiscreteModel)

  FOMStokesST(FEMInfo, model, get_g₀(FEMInfo))

end

struct FOMNavierStokesS{D} <: FOMS{D}
  FOMSS::FOMStokesS{D}
end

function FOMNavierStokesS(
  FEMInfo::FOMInfoS{3},
  model::DiscreteModel{D,D},
  g::Function) where D

  FOMNavierStokesS(FOMStokesS(FEMInfo, model, g))

end

function FOMNavierStokesS(
  FEMInfo::FOMInfoS{3},
  model::DiscreteModel)

  FOMNavierStokesS(FEMInfo, model, get_g₀(FEMInfo))

end

function Base.getproperty(FEMSpace::FOMNavierStokesS, sym::Symbol)
  getfield(FEMSpace.FOMSS, sym)
end

struct FOMNavierStokesST{D} <: FOMST{D}
  FOMSST::FOMStokesST{D}
end

function FOMNavierStokesST(
  FEMInfo::FOMInfoST{3},
  model::DiscreteModel{D,D},
  g::Function) where D

  FOMNavierStokesST(FOMStokesST(FEMInfo, model, g))

end

function FOMNavierStokesST(
  FEMInfo::FOMInfoST{3},
  model::DiscreteModel)

  FOMNavierStokesST(FEMInfo, model, get_g₀(FEMInfo))

end

function Base.getproperty(FEMSpace::FOMNavierStokesST, sym::Symbol)
  getfield(FEMSpace.FOMSST, sym)
end

abstract type ParamInfo end
abstract type ParamFormInfo end

mutable struct ParamInfoS <: ParamInfo
  μ::Vector{Float}
  var::String
  fun::Function
  θ::Vector{Vector{Float}}
end

mutable struct ParamInfoST <: ParamInfo
  μ::Vector{Float}
  var::String
  funₛ::Function
  funₜ::Function
  fun::Function
  θ::Vector{Vector{Float}}
end

function ParamInfo(
  FEMInfo::FOMInfoS{ID},
  μ::Vector,
  var::String) where ID

  fun = get_fun(FEMInfo, μ, var)
  ParamInfoS(μ, var, fun, Vector{Float}[])

end

function ParamInfo(
  FEMInfo::FOMInfoST{ID},
  μ::Vector,
  var::String) where ID

  funₛ, funₜ, fun  = get_fun(FEMInfo, μ, var)
  ParamInfoST(μ, var, funₛ, funₜ, fun, Vector{Float}[])

end

function ParamInfo(
  FEMInfo::FOMInfo{ID},
  μ::Vector{<:Vector},
  var::String) where ID

  Broadcasting(μᵢ -> ParamInfo(FEMInfo, μᵢ, var))(μ)

end

function ParamInfo(
  FEMInfo::FOMInfo{ID},
  μ::Vector,
  operators::Vector{String}) where ID

  get_single_ParamInfo(var) = ParamInfo(FEMInfo, μ, var)
  Broadcasting(get_single_ParamInfo)(operators)

end

function ParamInfo(
  Params::Vector{<:ParamInfo},
  var::String)

  for Param in Params
    if Param.var == var
      return Param
    end
  end

  error("Unrecognized variable")

end

function ParamInfo(
  Params::Vector{<:ParamInfo},
  vars::Vector{String})

  Broadcasting(var -> ParamInfo(Params, var))(vars)

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
  Param::ParamInfoS,
  dΩ::Measure)

  ParamFormInfoS(Param, dΩ)

end

function ParamFormInfo(
  Param::ParamInfoST,
  dΩ::Measure)

  ParamFormInfoST(Param, dΩ)

end

function ParamFormInfo(
  FEMSpace::FOM,
  Param::ParamInfo)

  ParamFormInfo(Param, get_measure(FEMSpace, Param.var))

end

function ParamFormInfo(
  FEMInfo::FOMInfo{ID},
  μ::Vector) where ID

  get_single_ParamFormInfo(var) = ParamInfo(FEMInfo, μ, var)
  operators = get_FEM_structures(FEMInfo)
  Broadcasting(get_single_ParamFormInfo)(operators)

end

function Base.getproperty(ParamForm::ParamFormInfoS, sym::Symbol)
  if sym in (:μ, :var, :fun, :θ)
    getfield(ParamForm.Param, sym)
  else
    getfield(ParamForm, sym)
  end
end

function Base.setproperty!(ParamForm::ParamFormInfoS, sym::Symbol, x::T) where T
  if sym in (:μ, :var, :fun, :θ)
    setfield!(ParamForm.Param, sym, x)::T
  else
    setfield!(ParamForm, sym, x)::T
  end
end

function Base.getproperty(ParamForm::ParamFormInfoST, sym::Symbol)
  if sym in (:μ, :var, :funₛ, :funₜ, :fun, :θ)
    getfield(ParamForm.Param, sym)
  else
    getfield(ParamForm, sym)
  end
end

function Base.setproperty!(ParamForm::ParamFormInfoST, sym::Symbol, x::T) where T
  if sym in (:μ, :var, :funₛ, :funₜ, :fun, :θ)
    setfield!(ParamForm.Param, sym, x)::T
  else
    setfield!(ParamForm, sym, x)::T
  end
end
