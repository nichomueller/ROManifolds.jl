function set_labels(probl::Info, model::DiscreteModel)

  labels = get_face_labeling(model)
  if !isempty(probl.dirichlet_tags) && !isempty(probl.dirichlet_bnds)
    for i = 1:length(probl.dirichlet_tags)
      if probl.dirichlet_tags[i] ∉ labels.tag_to_name
        add_tag_from_tags!(labels, probl.dirichlet_tags[i], probl.dirichlet_bnds[i])
      end
    end
  end
  if !isempty(probl.neumann_tags) && !isempty(probl.neumann_bnds)
    for i = 1:length(probl.neumann_tags)
      if probl.neumann_tags[i] ∉ labels.tag_to_name
        add_tag_from_tags!(labels, probl.neumann_tags[i], probl.neumann_bnds[i])
      end
    end
  end

  labels

end

function get_FESpace(::NTuple{1,Int}, probl::SteadyInfo, model::DiscreteModel, g = nothing)

  degree = 2*probl.order
  labels = set_labels(probl, model)
  Ω = Interior(model)
  dΩ = Measure(Ω, degree)
  Qₕ = CellQuadrature(Ω, degree)
  refFE = ReferenceFE(lagrangian, Float64, probl.order)
  if !isempty(probl.neumann_tags)
    Γn = BoundaryTriangulation(model, tags=probl.neumann_tags)
    dΓn = Measure(Γn, degree)
  else
    dΓn = nothing
  end
  if !isempty(probl.dirichlet_tags)
    Γd = BoundaryTriangulation(model, tags=probl.dirichlet_tags)
    dΓd = Measure(Γd, degree)
    V₀ = TestFESpace(model, refFE; conformity=:H1, dirichlet_tags=probl.dirichlet_tags, labels=labels)
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

  FE_space = FESpacePoisson(Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn)

  return FE_space

end

function get_FESpace(::NTuple{1,Int}, probl::UnsteadyInfo, model::DiscreteModel, g = nothing)

  degree = 2*probl.order
  labels = set_labels(probl, model)
  Ω = Interior(model)
  dΩ = Measure(Ω, degree)
  Qₕ = CellQuadrature(Ω, degree)
  refFE = ReferenceFE(lagrangian, Float64, probl.order)
  if !isempty(probl.neumann_tags)
    Γn = BoundaryTriangulation(model, tags=probl.neumann_tags)
    dΓn = Measure(Γn, degree)
  else
    dΓn = nothing
  end
  if !isempty(probl.dirichlet_tags)
    Γd = BoundaryTriangulation(model, tags=probl.dirichlet_tags)
    dΓd = Measure(Γd, degree)
    V₀ = TestFESpace(model, refFE; conformity=:H1, dirichlet_tags=probl.dirichlet_tags, labels=labels)
  else
    dΓd = nothing
    V₀ = TestFESpace(model, refFE; conformity=:H1, constraint=:zeromean)
  end
  if isnothing(g)
    g₀(x, t::Real) = 0
    g₀(t::Real) = x->g₀(x,t)
    V = TransientTrialFESpace(V₀, g₀)
  else
    V = TransientTrialFESpace(V₀, g)
  end
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  FE_space = FESpacePoissonUnsteady(Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn)

  return FE_space

end

function get_FESpace(::NTuple{2,Int}, probl::SteadyInfo, model::DiscreteModel, g = nothing)

  degree = 2*probl.order
  labels = set_labels(probl, model)

  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model, tags=probl.neumann_tags)
  dΓn = Measure(Γn, degree)
  Γd = BoundaryTriangulation(model, tags=probl.dirichlet_tags)
  dΓd = Measure(Γd, degree)
  Qₕ = CellQuadrature(Ω, degree)

  refFEᵤ = ReferenceFE(lagrangian, VectorValue{3,Float64}, probl.order)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1, dirichlet_tags=probl.dirichlet_tags, labels=labels)
  if !isnothing(g)
    V = TrialFESpace(V₀, g)
  else
    V = TrialFESpace(V₀, (x -> 0))
  end
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  refFEₚ = ReferenceFE(lagrangian, Float64, order-1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ; conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_trial_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q))

  FE_space = FESpaceStokes(Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn)

  return FE_space

end

function get_FESpace(::NTuple{2,Int}, probl::UnsteadyInfo, model::DiscreteModel, g = nothing)

  degree = 2*probl.order
  labels = set_labels(probl, model)

  Ω = Interior(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model, tags=probl.neumann_tags)
  dΓn = Measure(Γn, degree)
  Γd = BoundaryTriangulation(model, tags=probl.dirichlet_tags)
  dΓd = Measure(Γd, degree)
  Qₕ = CellQuadrature(Ω, degree)

  refFEᵤ = ReferenceFE(lagrangian, VectorValue{3,Float64}, probl.order)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1, dirichlet_tags=probl.dirichlet_tags, labels=labels)
  if isnothing(g)
    g₀(x, t::Real) = VectorValue(0,0,0)
    g₀(t::Real) = x->g₀(x,t)
    V = TransientTrialFESpace(V₀, g₀)
  else
    V = TransientTrialFESpace(V₀, g)
  end
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  refFEₚ = ReferenceFE(lagrangian, Float64, probl.order-1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ, conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q₀))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = TransientMultiFieldFESpace([V, Q])

  FE_space = FESpaceStokesUnsteady(Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn)

  return FE_space

end
