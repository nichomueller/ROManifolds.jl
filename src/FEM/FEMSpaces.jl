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

function get_FEMSpace(
  ::NTuple{1,Int}, ProblInfo::SteadyInfo, model::DiscreteModel, g=nothing)

  degree = 2 * ProblInfo.order
  labels = set_labels(ProblInfo, model)
  Ω = Interior(model)
  dΩ = Measure(Ω, degree)
  Qₕ = CellQuadrature(Ω, degree)
  refFE = ReferenceFE(lagrangian, Float64, ProblInfo.order)
  if !isempty(ProblInfo.neumann_tags)
    Γn = BoundaryTriangulation(model, tags=ProblInfo.neumann_tags)
    dΓn = Measure(Γn, degree)
  else
    dΓn = nothing
  end
  if !isempty(ProblInfo.dirichlet_tags)
    Γd = BoundaryTriangulation(model, tags=ProblInfo.dirichlet_tags)
    dΓd = Measure(Γd, degree)
    V₀ = TestFESpace(model, refFE; conformity=:H1,
      dirichlet_tags=ProblInfo.dirichlet_tags, labels=labels)
  else
    dΓd = nothing
    V₀ = TestFESpace(model, refFE; conformity=:H1, constraint=:zeromean)
  end
  if !isnothing(g)
    V = TrialFESpace(V₀, g)
  else
    V = TrialFESpace(V₀, (x -> 0))
  end

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  FEMSpace = FEMSpacePoissonSteady(Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{1,Int}, ProblInfo::UnsteadyInfo, model::DiscreteModel, g=nothing)

  degree = 2 * ProblInfo.order
  labels = set_labels(ProblInfo, model)
  Ω = Interior(model)
  dΩ = Measure(Ω, degree)
  Qₕ = CellQuadrature(Ω, degree)
  refFE = ReferenceFE(lagrangian, Float64, ProblInfo.order)
  if !isempty(ProblInfo.neumann_tags)
    Γn = BoundaryTriangulation(model, tags=ProblInfo.neumann_tags)
    dΓn = Measure(Γn, degree)
  else
    dΓn = nothing
  end
  if !isempty(ProblInfo.dirichlet_tags)
    Γd = BoundaryTriangulation(model, tags=ProblInfo.dirichlet_tags)
    dΓd = Measure(Γd, degree)
    V₀ = TestFESpace(model, refFE; conformity=:H1,
      dirichlet_tags=ProblInfo.dirichlet_tags, labels=labels)
  else
    dΓd = nothing
    V₀ = TestFESpace(model, refFE; conformity=:H1, constraint=:zeromean)
  end
  if isnothing(g)
    g₀(x, t::Real) = 0
    g₀(t::Real) = x -> g₀(x, t)
    V = TransientTrialFESpace(V₀, g₀)
  else
    V = TransientTrialFESpace(V₀, g)
  end
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  FEMSpace = FEMSpacePoissonUnsteady(Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{2,Int},
  ProblInfo::SteadyInfo,
  model::DiscreteModel,
  g=nothing)

  degree = 2 * ProblInfo.order
  labels = set_labels(ProblInfo, model)

  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model, tags=ProblInfo.neumann_tags)
  dΓn = Measure(Γn, degree)
  Γd = BoundaryTriangulation(model, tags=ProblInfo.dirichlet_tags)
  dΓd = Measure(Γd, degree)
  Qₕ = CellQuadrature(Ω, degree)

  refFEᵤ = ReferenceFE(lagrangian, VectorValue{3,Float64}, ProblInfo.order)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1,
    dirichlet_tags=ProblInfo.dirichlet_tags, labels=labels)
  if !isnothing(g)
    V = TrialFESpace(V₀, g)
  else
    V = TrialFESpace(V₀, (x -> 0))
  end
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  refFEₚ = ReferenceFE(lagrangian, Float64, order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ; conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_trial_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = TransientMultiFieldFESpace([V, Q])

  FEMSpace = FEMSpaceStokesSteady(
    Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn)

  return FEMSpace

end

function get_FEMSpace(
  ::NTuple{2,Int}, ProblInfo::UnsteadyInfo, model::DiscreteModel, g=nothing)

  degree = 2 * ProblInfo.order
  labels = set_labels(ProblInfo, model)
  Ω = Interior(model)
  dΩ = Measure(Ω, degree)
  Qₕ = CellQuadrature(Ω, degree)
  refFEᵤ = ReferenceFE(lagrangian, VectorValue{3,Float64}, ProblInfo.order)
  if !isempty(ProblInfo.neumann_tags)
    Γn = BoundaryTriangulation(model, tags=ProblInfo.neumann_tags)
    dΓn = Measure(Γn, degree)
  else
    dΓn = nothing
  end
  if !isempty(ProblInfo.dirichlet_tags)
    Γd = BoundaryTriangulation(model, tags=ProblInfo.dirichlet_tags)
    dΓd = Measure(Γd, degree)
    V₀ = TestFESpace(model, refFEᵤ; conformity=:H1,
      dirichlet_tags=ProblInfo.dirichlet_tags, labels=labels)
  else
    dΓd = nothing
    V₀ = TestFESpace(model, refFEᵤ; conformity=:H1, constraint=:zeromean)
  end
  if isnothing(g)
    g₀(x, t::Real) = VectorValue(0, 0, 0)
    g₀(t::Real) = x -> g₀(x, t)
    V = TransientTrialFESpace(V₀, g₀)
  else
    V = TransientTrialFESpace(V₀, g)
  end
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  refFEₚ = ReferenceFE(lagrangian, Float64, ProblInfo.order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ, conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q₀))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = TransientMultiFieldFESpace([V, Q])

  FEMSpace = FEMSpaceStokesUnsteady(
    Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn)

  return FEMSpace

end
