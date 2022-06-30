function set_labels(FEMInfo::Info, model::DiscreteModel)

  labels = get_face_labeling(model)
  if !isempty(FEMInfo.dirichlet_tags) && !isempty(FEMInfo.dirichlet_bnds)
    for i = eachindex(FEMInfo.dirichlet_tags)
      if FEMInfo.dirichlet_tags[i] ∉ labels.tag_to_name
        add_tag_from_tags!(labels, FEMInfo.dirichlet_tags[i], FEMInfo.dirichlet_bnds[i])
      end
    end
  end
  if !isempty(FEMInfo.neumann_tags) && !isempty(FEMInfo.neumann_bnds)
    for i = eachindex(FEMInfo.neumann_tags)
      if FEMInfo.neumann_tags[i] ∉ labels.tag_to_name
        add_tag_from_tags!(labels, FEMInfo.neumann_tags[i], FEMInfo.neumann_bnds[i])
      end
    end
  end

  labels

end

function get_mod_meas_quad(FEMInfo::Info, model::DiscreteModel)

  degree = 2 * FEMInfo.order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model, tags=FEMInfo.neumann_tags)
  dΓn = Measure(Γn, degree)
  Γd = BoundaryTriangulation(model, tags=FEMInfo.dirichlet_tags)
  dΓd = Measure(Γd, degree)
  Qₕ = CellQuadrature(Ω, degree)

  Ω, Qₕ, dΩ, dΓn, dΓd

end

function get_lagrangianQuad_info(
  ::SteadyInfo{T},
  model::DiscreteModel,
  Ω::BodyFittedTriangulation,
  Qₕ::CellQuadrature) where T

  ξₖ = get_cell_map(Ω)
  Qₕ_cell_point = get_cell_points(Qₕ)
  qₖ = get_data(Qₕ_cell_point)
  phys_quadp = lazy_map(evaluate,ξₖ,qₖ)
  refFE_quad = Gridap.ReferenceFE(lagrangianQuad,T,FEMInfo.order)
  V₀_quad = TestFESpace(model,refFE_quad,conformity=:L2)

  phys_quadp, V₀_quad

end

function get_lagrangianQuad_info(
  ::UnsteadyInfo{T},
  model::DiscreteModel,
  Ω::BodyFittedTriangulation,
  Qₕ::CellQuadrature) where T

  ξₖ = get_cell_map(Ω)
  Qₕ_cell_point = get_cell_points(Qₕ)
  qₖ = get_data(Qₕ_cell_point)
  phys_quadp = lazy_map(evaluate,ξₖ,qₖ)
  refFE_quad = Gridap.ReferenceFE(lagrangianQuad,T,FEMInfo.order)
  V₀_quad = TestFESpace(model,refFE_quad,conformity=:L2)

  phys_quadp, V₀_quad

end

function get_FEMSpace(
  ::NTuple{1,Int64},
  FEMInfo::SteadyInfo{T},
  model::DiscreteModel{D,D},
  g::F) where {D,T}

  Ω, Qₕ, dΩ, dΓn, dΓd = get_mod_meas_quad(FEMInfo, model)

  labels = set_labels(FEMInfo, model)
  refFE = Gridap.ReferenceFE(lagrangian, T, FEMInfo.order)

  V₀ = TestFESpace(model, refFE; conformity=:H1,
    dirichlet_tags=FEMInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  FEMSpace = FEMSpacePoissonSteady{D,T}(
    model, Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn, phys_quadp, V₀_quad)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{1,Int64},
  FEMInfo::UnsteadyInfo{T},
  model::DiscreteModel{D,D},
  g::F) where {D,T}

  Ω, Qₕ, dΩ, dΓn, dΓd = get_mod_meas_quad(FEMInfo, model)

  refFE = Gridap.ReferenceFE(lagrangian, T, FEMInfo.order)
  labels = set_labels(FEMInfo, model)
  V₀ = TestFESpace(model, refFE; conformity=:H1,
    dirichlet_tags=FEMInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  FEMSpace = FEMSpacePoissonUnsteady{D,T}(
    model, Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn, phys_quadp, V₀_quad)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{2,Int64},
  FEMInfo::SteadyInfo{T},
  model::DiscreteModel{D,D},
  g::F) where {D,T}

  Ω, Qₕ, dΩ, dΓn, dΓd = get_mod_meas_quad(FEMInfo, model)

  refFEᵤ = Gridap.ReferenceFE(lagrangian, VectorValue{D,T}, FEMInfo.order)
  labels = set_labels(FEMInfo, model)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1,
    dirichlet_tags=FEMInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  refFEₚ = Gridap.ReferenceFE(lagrangian, T, order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ; conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_trial_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = TransientMultiFieldFESpace([V, Q])

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  FEMSpace = FEMSpaceStokesSteady{D,T}(
    model, Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn,
    phys_quadp, V₀_quad)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{2,Int64},
  FEMInfo::UnsteadyInfo{T},
  model::DiscreteModel{D,D},
  g::F) where {D,T}

  Ω, Qₕ, dΩ, dΓn, dΓd = get_mod_meas_quad(FEMInfo, model)

  refFEᵤ = Gridap.ReferenceFE(lagrangian, VectorValue{D,T}, FEMInfo.order)
  labels = set_labels(FEMInfo, model)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1,
    dirichlet_tags=FEMInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  refFEₚ = Gridap.ReferenceFE(lagrangian, T, FEMInfo.order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ, conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q₀))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = TransientMultiFieldFESpace([V, Q])

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  FEMSpace = FEMSpaceStokesUnsteady{D,T}(
    model, Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn,
    phys_quadp, V₀_quad)

  return FEMSpace

end

function get_FEMSpace₀(
  problem_id::NTuple{1,Int64},
  FEMInfo::SteadyInfo,
  model::DiscreteModel)

  get_FEMSpace(problem_id,FEMInfo,model,x->0)

end

function get_FEMSpace₀(
  problem_id::NTuple{1,Int64},
  FEMInfo::UnsteadyInfo{T},
  model::DiscreteModel) where T

  g₀(x, t::Real) = zero(T)
  g₀(t::Real) = x -> g₀(x, t)
  get_FEMSpace(problem_id,FEMInfo,model,g₀)

end

function get_FEMSpace₀(
  problem_id::NTuple{2,Int64},
  FEMInfo::SteadyInfo,
  model::DiscreteModel)

  get_FEMSpace(problem_id,FEMInfo,model,x->0)

end

function get_FEMSpace₀(
  problem_id::NTuple{2,Int64},
  FEMInfo::UnsteadyInfo{T},
  model::DiscreteModel,) where T

  g₀(x, t::Real) = zero(T)
  g₀(t::Real) = x -> g₀(x, t)
  get_FEMSpace(problem_id,FEMInfo,model,g₀)

end
