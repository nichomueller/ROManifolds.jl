function get_mod_meas_quad(FEMInfo::FOMInfo, model::DiscreteModel)

  set_labels(model, FEMInfo.bnd_info)

  degree = 2 * FEMInfo.order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model, tags=["neumann"])
  dΓn = Measure(Γn, degree)::Measure
  Qₕ = CellQuadrature(Ω, degree)::CellQuadrature

  Ω, Γn, Qₕ, dΩ, dΓn

end

function get_lagrangianQuad_info(
  FEMInfo::FOMInfo,
  model::DiscreteModel{Dc,Dp},
  Ω::BodyFittedTriangulation,
  Qₕ::CellQuadrature) where {Dc,Dp}

  ξₖ = get_cell_map(Ω)
  Qₕ_cell_point = get_cell_points(Qₕ)
  qₖ = get_data(Qₕ_cell_point)
  phys_quadp = collect(lazy_map(Gridap.evaluate,ξₖ,qₖ))::Vector{Vector{VectorValue{Dp,Float}}}
  refFE_quad = Gridap.ReferenceFE(lagrangianQuad,Float,FEMInfo.order)
  V₀_quad = TestFESpace(model,refFE_quad,conformity=:L2)

  phys_quadp, V₀_quad

end

function FEMSpace_quantities(
  ::NTuple{1,Int},
  FEMInfo::FOMInfoS,
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

  Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, Γn, dΩ, dΓn, phys_quadp, V₀_quad

end

function FEMSpace_quantities(
  ::NTuple{1,Int},
  FEMInfo::FOMInfoST,
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

  Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, Γn, dΩ, dΓn, phys_quadp, V₀_quad

end

function FEMSpace_quantities(
  ::NTuple{2,Int},
  FEMInfo::FOMInfoS,
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

  Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, Γn, dΩ, dΓn,
    phys_quadp, V₀_quad

end

function FEMSpace_quantities(
  ::NTuple{2,Int},
  FEMInfo::FOMInfoST,
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

  Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, Γn, dΩ, dΓn,
    phys_quadp, V₀_quad

end

function FEMSpace_quantities(
  ::NTuple{3,Int},
  FEMInfo::FOMInfoS,
  model::DiscreteModel{D,D},
  g::Function) where D

  FEMSpace_quantities(NTuple(2, Int), FEMInfo, model, g)

end

function FEMSpace_quantities(
  ::NTuple{3,Int},
  FEMInfo::FOMInfoST,
  model::DiscreteModel{D,D},
  g::Function) where D

  FEMSpace_quantities(NTuple(2, Int), FEMInfo, model, g)

end

function FEMSpace(
  NT::NTuple{1,Int},
  FEMInfo::FOMInfoS,
  model::DiscreteModel{D,D},
  g::Function) where D

  FOMPoissonS{D}(
    model, FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function FEMSpace(
  NT::NTuple{1,Int},
  FEMInfo::FOMInfoST,
  model::DiscreteModel{D,D},
  g::Function) where D

  FOMPoissonST{D}(
    model, FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function FEMSpace(
  NT::NTuple{2,Int},
  FEMInfo::FOMInfoS,
  model::DiscreteModel{D,D},
  g::Function) where D

  FOMStokesS{D}(
    model, FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function FEMSpace(
  NT::NTuple{2,Int},
  FEMInfo::FOMInfoST,
  model::DiscreteModel{D,D},
  g::Function) where D

  FOMStokesST{D}(
    model, FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function FEMSpace(
  NT::NTuple{3,Int},
  FEMInfo::FOMInfoS,
  model::DiscreteModel{D,D},
  g::Function) where D

  FOMNavierStokesS{D}(
    model, FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function FEMSpace(
  NT::NTuple{3,Int},
  FEMInfo::FOMInfoST,
  model::DiscreteModel{D,D},
  g::Function) where D

  FOMNavierStokesST{D}(
    model, FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function FEMSpace₀(
  id::NTuple{1,Int},
  FEMInfo::FOMInfoS,
  model::DiscreteModel)

  FEMSpace(id,FEMInfo,model,x->0)

end

function FEMSpace₀(
  id::NTuple{1,Int},
  FEMInfo::FOMInfoST,
  model::DiscreteModel)

  g₀(x, t::Real) = 0.
  g₀(t::Real) = x -> g₀(x, t)
  FEMSpace(id,FEMInfo,model,g₀)

end

function FEMSpace₀(
  id::NTuple{2,Int},
  FEMInfo::FOMInfoS,
  model::DiscreteModel)

  FEMSpace(id,FEMInfo,model, x->zero(VectorValue(FEMInfo.D, Float)))

end

function FEMSpace₀(
  id::NTuple{2,Int},
  FEMInfo::FOMInfoST,
  model::DiscreteModel)

  g₀(x, t::Real) = zero(VectorValue(FEMInfo.D, Float))
  g₀(t::Real) = x -> g₀(x, t)
  FEMSpace(id,FEMInfo,model,g₀)

end

function FEMSpace₀(
  id::NTuple{3,Int},
  FEMInfo::FOMInfoS,
  model::DiscreteModel)

  FEMSpace(id,FEMInfo,model, x->zero(VectorValue(FEMInfo.D, Float)))

end

function FEMSpace₀(
  id::NTuple{3,Int},
  FEMInfo::FOMInfoST,
  model::DiscreteModel)

  g₀(x, t::Real) = zero(VectorValue(FEMInfo.D, Float))
  g₀(t::Real) = x -> g₀(x, t)
  FEMSpace(id,FEMInfo,model,g₀)

end

function FEMSpace_vectors(FEMSpace::FOM, var::String)
  if var == "Lc"
    FEMSpace.Q₀
  else
    FEMSpace.V₀
  end
end

function FEMSpace_matrices(FEMSpace::FOM, var::String)
  if var == "B"
    FEMSpace.V, FEMSpace.Q₀
  else
    FEMSpace.V, FEMSpace.V₀
  end
end

function get_measure(FEMSpace::FOM, var::String)
  if var == "H"
    FEMSpace.dΓn
  else
    FEMSpace.dΩ
  end
end
