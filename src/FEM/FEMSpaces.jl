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

  FESpace = FESpacePoissonSteady(Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn)

  return FESpace

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

  FESpace = FESpacePoissonUnsteady(Qₕ, V₀, V, ϕᵥ, ϕᵤ, Nₛᵘ, Ω, dΩ, dΓd, dΓn)

  return FESpace

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

  FESpace = FESpaceStokes(Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn)

  return FESpace

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

  FESpace = FESpaceStokesUnsteady(Qₕ, V₀, V, Q₀, Q, X₀, X, ϕᵥ, ϕᵤ, ψᵧ, ψₚ, Nₛᵘ, Nₛᵖ, Ω, dΩ, Γd, dΓd, dΓn)

  return FESpace

end

abstract type LagrangianQuadRefFE{D} <: LagrangianRefFE{D} end

struct LagrangianQuad <: ReferenceFEName end

const lagrangianQuad = LagrangianQuad()

function ReferenceFE(
  polytope::Polytope,
  ::LagrangianQuad,
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}}) where T
  LagrangianQuadRefFE(T,polytope,orders)
end

function LagrangianQuadRefFE(
  ::Type{T},
  p::Polytope{D},
  order::Int) where {T,D}
  orders = tfill(order,Val{D}())
  LagrangianQuadRefFE(T,p,orders)
end

function LagrangianQuadRefFE(
  ::Type{T},
  p::Polytope{D},
  orders) where {T,D}
  _lagrangian_quad_ref_fe(T,p,orders)
end

function _lagrangian_quad_ref_fe(::Type{T},
  p::Polytope{D},
  orders) where {T,D}

  @assert isa(p,ExtrusionPolytope)
  @assert is_n_cube(p)
  degrees= broadcast(*,2,orders)
  q=Quadrature(p,Gridap.ReferenceFEs.TensorProduct(),degrees)
  nodes = get_coordinates(q)

  prebasis = compute_monomial_basis(T,p,orders)

  # Compute face_own_nodes
  face_nodes = [Int[] for i in 1:num_faces(p)]
  push!(last(face_nodes),collect(1:length(nodes))...)

  # Compute face_own_nodes
  face_dofs = [Int[] for i in 1:num_faces(p)]
  push!(last(face_dofs),collect(1:length(nodes)*num_components(T))...)

  dofs = LagrangianDofBasis(T,nodes)

  nnodes = length(dofs.nodes)
  ndofs = length(dofs.dof_to_node)
  metadata = nothing

  conf = L2Conformity()
  reffe = GenericRefFE{typeof(conf)}(
    ndofs,
    p,
    prebasis,
    dofs,
    conf,
    metadata,
    face_dofs)
  GenericLagrangianRefFE(reffe,face_nodes)
end

function tfill(v, ::Val{D}) where D
  t = tfill(v, Val{D-1}())
  (v,t...)
end

tfill(v,::Val{0}) = ()
tfill(v,::Val{1}) = (v,)
tfill(v,::Val{2}) = (v,v)
tfill(v,::Val{3}) = (v,v,v)
