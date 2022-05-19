abstract type LagrangianQuadRefFE{D} <: LagrangianRefFE{D} end

struct LagrangianQuad <: ReferenceFEName end

const lagrangian_quad = LagrangianQuad()

function LagrangianQuadRefFE(
  ::Type{T},
  p::Polytope{D},
  orders,
  trian::Gridap.Geometry.BodyFittedTriangulation;
  space::Symbol=_default_space(p)) where {T,D}

  if space == :P && is_n_cube(p)
    return _PDiscRefFE(T,p,orders)
  elseif space == :S && is_n_cube(p)
    SerendipityRefFE(T,p,orders)
  else
    if any(map(i->i==0,orders)) && !all(map(i->i==0,orders))
      cont = map(i -> i == 0 ? DISC : CONT,orders)
      return _cd_lagrangian_quad_ref_fe(T,p,orders,trian,cont)
    else
      return _lagrangian_quad_ref_fe(T,p,orders,trian)
    end
  end
end

function ReferenceFE(
  polytope::Polytope,
  ::LagrangianQuad,
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}};
  space::Symbol=_default_space(polytope)) where T

  LagrangianQuadRefFE(T,polytope,orders;space=space)
end

function _lagrangian_quad_ref_fe(::Type{T},
  p::Polytope{D},
  orders,
  trian::Gridap.Geometry.BodyFittedTriangulation) where {T,D}

  prebasis = compute_monomial_basis(T,p,orders)
  Qₕ = CellQuadrature(trian, order[1])
  Qₕ_cell_data = get_data(Qₕ)
  quad = Qₕ_cell_data[rand(1:num_cells(trian))]
  quad_pts = get_coordinates(quad)

  #nodes, face_own_nodes = compute_nodes(p,orders)
  dofs = LagrangianDofBasis(T,quad_pts)
  reffaces = compute_lagrangian_reffaces(T,p,orders)

  #nnodes = length(dofs.nodes)
  ndofs = length(dofs.dof_to_node)
  metadata = reffaces
  #_reffaces = vcat(reffaces...)
  face_quads = Vector{Int}[]
  [push!(face_quads, Int[]) for node = 1:ndofs-1]
  push!(face_quads, collect(1:ndofs))
  face_dofs = face_quads
  #= face_nodes = _generate_face_nodes(nnodes,face_own_nodes,p,_reffaces)
  face_own_dofs = _generate_face_own_dofs(face_own_nodes, dofs.node_and_comp_to_dof)
  face_dofs = _generate_face_dofs(ndofs,face_own_dofs,p,_reffaces) =#

  if all(map(i->i==0,orders) ) && D>0
    conf = L2Conformity()
  else
    conf = GradConformity()
  end

  reffe = GenericRefFE{typeof(conf)}(
    ndofs,
    p,
    prebasis,
    dofs,
    conf,
    metadata,
    face_dofs)

  GenericLagrangianRefFE(reffe,face_quads)

end

function _cd_lagrangian_quad_ref_fe(
  ::Type{T},
  p::ExtrusionPolytope{D},
  orders,
  trian::Gridap.Geometry.BodyFittedTriangulation,
  cont) where {T,D}

  @assert isa(p,ExtrusionPolytope)

  prebasis = compute_monomial_basis(T,p,orders)
  Qₕ = CellQuadrature(trian, order[1])
  Qₕ_cell_data = get_data(Qₕ)
  quad = Qₕ_cell_data[rand(1:num_cells(trian))]
  quad_pts = get_coordinates(quad)

  #nodes, face_own_nodes = cd_compute_nodes(p,orders)
  dofs = LagrangianDofBasis(T,quad_pts)

  #nnodes = length(dofs.nodes)
  ndofs = length(dofs.dof_to_node)

  #= face_own_nodes = _compute_cd_face_own_nodes(p,orders,cont)
  face_nodes = _compute_face_nodes(p,face_own_nodes)

  face_own_dofs = _generate_face_own_dofs(face_own_nodes, dofs.node_and_comp_to_dof)
  face_dofs = _compute_face_nodes(p,face_own_dofs) =#

  face_quads = Vector{Int}[]
  [push!(face_quads, Int[]) for node = 1:ndofs-1]
  push!(face_quads, collect(1:ndofs))
  face_dofs = face_quads

  data = nothing

  conf = CDConformity(Tuple(cont))

  reffe = GenericRefFE{typeof(conf)}(
      ndofs,
      p,
      prebasis,
      dofs,
      conf,
      data,
      face_dofs)

  GenericLagrangianRefFE(reffe,face_quads)
end

function _generate_face_quad_dofs(ndofs,face_to_own_dofs,polytope,reffaces)

  face_to_num_fdofs = map(num_dofs,reffaces)
  push!(face_to_num_fdofs,ndofs)

  face_to_lface_to_own_fdofs = map(get_face_own_dofs,reffaces)
  push!(face_to_lface_to_own_fdofs,face_to_own_dofs)

  face_to_lface_to_face = get_faces(polytope)

_generate_face_nodes_aux(
  ndofs,
  face_to_own_dofs,
  face_to_num_fdofs,
  face_to_lface_to_own_fdofs,
  face_to_lface_to_face)
end
