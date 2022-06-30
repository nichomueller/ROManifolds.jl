function set_labels(ProblInfo::Info, model::DiscreteModel)

  labels = get_face_labeling(model)
  if !isempty(ProblInfo.dirichlet_tags) && !isempty(ProblInfo.dirichlet_bnds)
    for i = eachindex(ProblInfo.dirichlet_tags)
      if ProblInfo.dirichlet_tags[i] ∉ labels.tag_to_name
        add_tag_from_tags!(labels, ProblInfo.dirichlet_tags[i], ProblInfo.dirichlet_bnds[i])
      end
    end
  end
  if !isempty(ProblInfo.neumann_tags) && !isempty(ProblInfo.neumann_bnds)
    for i = eachindex(ProblInfo.neumann_tags)
      if ProblInfo.neumann_tags[i] ∉ labels.tag_to_name
        add_tag_from_tags!(labels, ProblInfo.neumann_tags[i], ProblInfo.neumann_bnds[i])
      end
    end
  end

  labels

end

function get_measures_quadrature(
  ProblInfo::Info,
  model::DiscreteModel)

  degree = 2 * ProblInfo.order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model, tags=ProblInfo.neumann_tags)
  dΓn = Measure(Γn, degree)
  Γd = BoundaryTriangulation(model, tags=ProblInfo.dirichlet_tags)
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
  ProblInfo::SteadyInfo{T},
  model::DiscreteModel{D,D},
  g::F) where {D,T}

  Ω, Qₕ, dΩ, dΓn, dΓd = get_measures_quadrature(ProblInfo, model)

  labels = set_labels(ProblInfo, model)
  refFE = Gridap.ReferenceFE(lagrangian, T, ProblInfo.order)

  V₀ = TestFESpace(model, refFE; conformity=:H1,
    dirichlet_tags=ProblInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  phys_quadp, V₀_quad = get_lagrangianQuad_info(ProblInfo, model, Ω, Qₕ)

  FEMSpace = FEMSpacePoissonSteady{D,T}(
    Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn, phys_quadp, V₀_quad)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{1,Int64},
  ProblInfo::UnsteadyInfo{T},
  model::DiscreteModel{D,D},
  g::F) where {D,T}

  Ω, Qₕ, dΩ, dΓn, dΓd = get_measures_quadrature(ProblInfo, model)

  refFE = Gridap.ReferenceFE(lagrangian, T, ProblInfo.order)
  labels = set_labels(ProblInfo, model)
  V₀ = TestFESpace(model, refFE; conformity=:H1,
    dirichlet_tags=ProblInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  phys_quadp, V₀_quad = get_lagrangianQuad_info(ProblInfo, model, Ω, Qₕ)

  FEMSpace = FEMSpacePoissonUnsteady{D,T}(
    Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn, phys_quadp, V₀_quad)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{2,Int64},
  ProblInfo::SteadyInfo{T},
  model::DiscreteModel{D,D},
  g::F) where {D,T}

  Ω, Qₕ, dΩ, dΓn, dΓd = get_measures_quadrature(ProblInfo, model)

  refFEᵤ = Gridap.ReferenceFE(lagrangian, VectorValue{D,T}, ProblInfo.order)
  labels = set_labels(ProblInfo, model)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1,
    dirichlet_tags=ProblInfo.dirichlet_tags, labels=labels)
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

  phys_quadp, V₀_quad = get_lagrangianQuad_info(ProblInfo, model, Ω, Qₕ)

  FEMSpace = FEMSpaceStokesSteady{D,T}(
    Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn,
    phys_quadp, V₀_quad)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{2,Int64},
  ProblInfo::UnsteadyInfo{T},
  model::DiscreteModel{D,D},
  g::F) where {D,T}

  Ω, Qₕ, dΩ, dΓn, dΓd = get_measures_quadrature(ProblInfo, model)

  refFEᵤ = Gridap.ReferenceFE(lagrangian, VectorValue{D,T}, ProblInfo.order)
  labels = set_labels(ProblInfo, model)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1,
    dirichlet_tags=ProblInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  refFEₚ = Gridap.ReferenceFE(lagrangian, T, ProblInfo.order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ, conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q₀))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = TransientMultiFieldFESpace([V, Q])

  phys_quadp, V₀_quad = get_lagrangianQuad_info(ProblInfo, model, Ω, Qₕ)

  FEMSpace = FEMSpaceStokesUnsteady{D,T}(
    Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn,
    phys_quadp, V₀_quad)

  return FEMSpace

end

function get_FEMSpace₀(
  problem_id::NTuple{1,Int64},
  ProblInfo::SteadyInfo,
  model::DiscreteModel)

  get_FEMSpace(problem_id,ProblInfo,model,x->0)

end

function get_FEMSpace₀(
  problem_id::NTuple{1,Int64},
  ProblInfo::UnsteadyInfo{T},
  model::DiscreteModel) where T

  g₀(x, t::Real) = zero(T)
  g₀(t::Real) = x -> g₀(x, t)
  get_FEMSpace(problem_id,ProblInfo,model,g₀)

end

function get_FEMSpace₀(
  problem_id::NTuple{2,Int64},
  ProblInfo::SteadyInfo,
  model::DiscreteModel)

  get_FEMSpace(problem_id,ProblInfo,model,x->0)

end

function get_FEMSpace₀(
  problem_id::NTuple{2,Int64},
  ProblInfo::UnsteadyInfo{T},
  model::DiscreteModel) where T

  g₀(x, t::Real) = zero(T)
  g₀(t::Real) = x -> g₀(x, t)
  get_FEMSpace(problem_id,ProblInfo,model,g₀)

end
