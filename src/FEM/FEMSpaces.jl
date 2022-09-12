function get_mod_meas_quad(FEMInfo::Info, model::DiscreteModel)

  degree = 2 * FEMInfo.order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γn = BoundaryTriangulation(model, tags=FEMInfo.neumann_tags)
  dΓn = Measure(Γn, degree)::Measure
  Γd = BoundaryTriangulation(model, tags=FEMInfo.dirichlet_tags)
  dΓd = Measure(Γd, degree)::Measure
  Qₕ = CellQuadrature(Ω, degree)::CellQuadrature

  Ω, Γn, Qₕ, dΩ, dΓn, dΓd

end

function get_lagrangianQuad_info(
  FEMInfo::Info,
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

function get_FEMSpace_quantities(
  ::NTuple{1,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  Ω, Γn, Qₕ, dΩ, dΓn, dΓd = get_mod_meas_quad(FEMInfo, model)

  labels = set_labels(FEMInfo, model)
  refFE = Gridap.ReferenceFE(lagrangian, Float, FEMInfo.order)

  V₀ = TestFESpace(model, refFE; conformity=:H1,
    dirichlet_tags=FEMInfo.dirichlet_tags, labels=labels)
  V = TrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, Γn, dΩ, dΓd, dΓn, phys_quadp, V₀_quad

end

function get_FEMSpace_quantities(
  ::NTuple{1,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  Ω, Γn, Qₕ, dΩ, dΓn, dΓd = get_mod_meas_quad(FEMInfo, model)

  refFE = Gridap.ReferenceFE(lagrangian, Float, FEMInfo.order)
  labels = set_labels(FEMInfo, model)
  V₀ = TestFESpace(model, refFE; conformity=:H1,
    dirichlet_tags=FEMInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, Γn, dΩ, dΓd, dΓn, phys_quadp, V₀_quad

end

function get_FEMSpace_quantities(
  ::NTuple{2,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  get_FEMSpace_quantities(get_NTuple(1, Int), FEMInfo, model, g)

end

function get_FEMSpace_quantities(
  ::NTuple{2,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  get_FEMSpace_quantities(get_NTuple(1, Int), FEMInfo, model, g)

end

function get_FEMSpace_quantities(
  ::NTuple{3,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  Ω, Γn, Qₕ, dΩ, dΓn, dΓd = get_mod_meas_quad(FEMInfo, model)

  refFEᵤ = Gridap.ReferenceFE(lagrangian, VectorValue{D,Float}, FEMInfo.order)
  labels = set_labels(FEMInfo, model)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1,
    dirichlet_tags=FEMInfo.dirichlet_tags, labels=labels)
  V = TrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  Nₛᵘ = length(get_free_dof_ids(V))

  refFEₚ = Gridap.ReferenceFE(lagrangian, Float, order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ; conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = MultiFieldFESpace([V, Q])

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, Γn, dΩ, dΓd, dΓn,
    phys_quadp, V₀_quad

end

function get_FEMSpace_quantities(
  ::NTuple{3,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  Ω, Γn, Qₕ, dΩ, dΓn, dΓd = get_mod_meas_quad(FEMInfo, model)

  refFEᵤ = Gridap.ReferenceFE(lagrangian, VectorValue{D,Float}, FEMInfo.order)
  labels = set_labels(FEMInfo, model)
  V₀ = TestFESpace(model, refFEᵤ; conformity=:H1,
    dirichlet_tags=FEMInfo.dirichlet_tags, labels=labels)
  V = TransientTrialFESpace(V₀, g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  Nₛᵘ = length(get_free_dof_ids(V₀))

  refFEₚ = Gridap.ReferenceFE(lagrangian, Float, FEMInfo.order - 1; space=:P)
  Q₀ = TestFESpace(model, refFEₚ, conformity=:L2, constraint=:zeromean)
  Q = TrialFESpace(Q₀)
  ψᵧ = get_fe_basis(Q₀)
  ψₚ = get_trial_fe_basis(Q)
  Nₛᵖ = length(get_free_dof_ids(Q₀))

  X₀ = MultiFieldFESpace([V₀, Q₀])
  X = TransientMultiFieldFESpace([V, Q])

  phys_quadp, V₀_quad = get_lagrangianQuad_info(FEMInfo, model, Ω, Qₕ)

  Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, Γn, dΩ, dΓd, dΓn,
    phys_quadp, V₀_quad

end

function get_FEMSpace_quantities(
  ::NTuple{4,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  get_FEMSpace_quantities(get_NTuple(3, Int), FEMInfo, model, g)

end

function get_FEMSpace_quantities(
  ::NTuple{4,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  get_FEMSpace_quantities(get_NTuple(3, Int), FEMInfo, model, g)

end

function get_FEMSpace(
  NT::NTuple{1,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  return FEMSpacePoissonSteady{D}(
    model, get_FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function get_FEMSpace(
  NT::NTuple{1,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  return FEMSpacePoissonUnsteady{D}(
    model, get_FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function get_FEMSpace(
  NT::NTuple{2,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  return FEMSpaceADRSteady{D}(
    model, get_FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function get_FEMSpace(
  NT::NTuple{2,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  return FEMSpaceADRUnsteady{D}(
    model, get_FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function get_FEMSpace(
  NT::NTuple{3,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  return FEMSpaceStokesSteady{D}(
    model, get_FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function get_FEMSpace(
  NT::NTuple{3,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  return FEMSpaceStokesUnsteady{D}(
    model, get_FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function get_FEMSpace(
  NT::NTuple{4,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  return FEMSpaceNavierStokesSteady{D}(
    model, get_FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function get_FEMSpace(
  NT::NTuple{4,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel{D,D},
  g::F) where D

  return FEMSpaceNavierStokesUnsteady{D}(
    model, get_FEMSpace_quantities(NT, FEMInfo, model, g)...)

end

function get_FEMSpace₀(
  problem_id::NTuple{1,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel)

  get_FEMSpace(problem_id,FEMInfo,model,x->0)

end

function get_FEMSpace₀(
  problem_id::NTuple{1,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel)

  g₀(x, t::Real) = 0.
  g₀(t::Real) = x -> g₀(x, t)
  get_FEMSpace(problem_id,FEMInfo,model,g₀)

end

function get_FEMSpace₀(
  problem_id::NTuple{2,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel)

  get_FEMSpace(problem_id,FEMInfo,model,x->0)

end

function get_FEMSpace₀(
  problem_id::NTuple{2,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel)

  g₀(x, t::Real) = 0.
  g₀(t::Real) = x -> g₀(x, t)
  get_FEMSpace(problem_id,FEMInfo,model,g₀)

end

function get_FEMSpace₀(
  problem_id::NTuple{3,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel)

  get_FEMSpace(problem_id,FEMInfo,model, x->zero(VectorValue(FEMInfo.D, Float)))

end

function get_FEMSpace₀(
  problem_id::NTuple{3,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel)

  g₀(x, t::Real) = zero(VectorValue(FEMInfo.D, Float))
  g₀(t::Real) = x -> g₀(x, t)
  get_FEMSpace(problem_id,FEMInfo,model,g₀)

end

function get_FEMSpace₀(
  problem_id::NTuple{4,Int},
  FEMInfo::SteadyInfo,
  model::DiscreteModel)

  get_FEMSpace(problem_id,FEMInfo,model, x->zero(VectorValue(FEMInfo.D, Float)))

end

function get_FEMSpace₀(
  problem_id::NTuple{4,Int},
  FEMInfo::UnsteadyInfo,
  model::DiscreteModel)

  g₀(x, t::Real) = zero(VectorValue(FEMInfo.D, Float))
  g₀(t::Real) = x -> g₀(x, t)
  get_FEMSpace(problem_id,FEMInfo,model,g₀)

end
