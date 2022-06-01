abstract type LagrangianQuadRefFE{D} <: LagrangianRefFE{D} end

struct LagrangianQuad <: ReferenceFEName end

const lagrangian_quad = LagrangianQuad()

function ReferenceFE(
  polytope::Polytope,
  ::LagrangianQuad,
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}},
  trian::Gridap.Geometry.BodyFittedTriangulation;
  space::Symbol=_default_space(polytope)) where T

  LagrangianQuadRefFE(T,polytope,orders,trian;space=space)
end

function LagrangianQuadRefFE(
  ::Type{T},
  p::Polytope{D},
  order::Int,
  trian::Gridap.Geometry.BodyFittedTriangulation;
  space::Symbol=_default_space(p)) where {T,D}

  orders = tfill(order,Val{D}())
  LagrangianQuadRefFE(T,p,orders,trian;space=space)
end

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

function _lagrangian_quad_ref_fe(::Type{T},
  p::Polytope{D},
  orders,
  trian::Gridap.Geometry.BodyFittedTriangulation) where {T,D}

  degree = 2
  Qₕ = CellQuadrature(trian, degree)
  Qₕ_cell_data = get_data(Qₕ)
  quad = Qₕ_cell_data[rand(1:num_cells(trian))]
  quad_pts = get_coordinates(quad)

  prebasis = compute_monomial_basis(T,p,orders)
  nodes, face_own_nodes = my_compute_nodes(p,orders,quad_pts)
  dofs = LagrangianDofBasis(T,nodes)
  reffaces = compute_lagrangian_quad_reffaces(T,p,orders,trian)

  nnodes = length(dofs.nodes)
  ndofs = length(dofs.dof_to_node)
  metadata = reffaces
  _reffaces = vcat(reffaces...)
  face_nodes = _generate_face_nodes(nnodes,face_own_nodes,p,_reffaces)
  face_own_dofs = _generate_face_own_dofs(face_own_nodes, dofs.node_and_comp_to_dof)
  face_dofs = _generate_face_dofs(ndofs,face_own_dofs,p,_reffaces)

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

  GenericLagrangianRefFE(reffe,face_nodes)

end

function my_compute_nodes(p::ExtrusionPolytope{D},orders,quad_pts) where D
  nodes, facenodes = _compute_nodes(p,orders,quad_pts)
  if any(map(i->i==0,orders))
    return (nodes, facenodes)
  end
  #terms = _coords_to_terms(_nodes,orders)
  #nodes = _terms_to_coords(terms,orders)
  (nodes, facenodes)
end

function _compute_nodes(p,orders,quad_pts)
  if any( map(i->i==0,orders))
    return _compute_constant_nodes(p,orders,quad_pts)
  elseif all(map(i->i==1,orders))
    return _compute_linear_nodes(p,quad_pts)
  else
    error("Not implemented higher order yet")
  end
end

function _compute_constant_nodes(p,orders,quad_pts)
  @assert all( orders .== 0) "If an order is 0 in some direction, it should be 0 also in the others"
  x = compute_own_nodes(p,orders,quad_pts)
  facenodes = [Int[] for i in 1:num_faces(p)]
  push!(facenodes[end],1)
  x, facenodes
end

function compute_own_nodes(p::ExtrusionPolytope{D},orders,quad_pts) where D
  extrusion = Tuple(p.extrusion)
  if all(map(i->i==0,orders))
    _interior_nodes_order_0(p,quad_pts)
  else
    error("Not implemented higher order yet")
  end
end

function _interior_nodes_order_0(p,quad_pts)
  x = get_nodes(p,quad_pts)
  x0 = sum(x) / length(x)
  [x0,]
end

function _compute_linear_nodes(p,quad_pts)
  x = get_nodes(p,quad_pts)
  facenodes = [Int[] for i in 1:num_faces(p)]
  for i in 1:num_vertices(p)
    push!(facenodes[i],i)
  end
  x, facenodes
end

function _coords_to_terms(coords::Vector{<:Point{D}},orders) where D
  indexbase = 1
  terms = CartesianIndex{D}[]
  P = Point{D,Int}
  t = zero(Mutable(P))
  for x in coords
    for d in 1:D
      t[d] = round(x[d]*orders[d]) + indexbase
    end
    term = CartesianIndex(Tuple(t))
    push!(terms,term)
  end
  terms
end

function  _terms_to_coords(terms::Vector{CartesianIndex{D}},orders) where D
  P = Point{D,Float64}
  indexbase = 1
  nodes = P[]
  x = zero(Mutable(P))
  for t in terms
    for d in 1:D
      x[d] = (t[d] - indexbase) / orders[d]
    end
    node = P(x)
    push!(nodes,node)
  end
  nodes
end

function get_nodes(p::Polytope{D},quad_pts::Vector) where {D}
  if D == 0
    return get_vertex_coordinates(p)
  elseif D == 1
    return [Point(quad_pts[1][1]),Point(quad_pts[2][1])]
  elseif p == TRI
    return [Point(quad_pts[1][1],quad_pts[1][2]), Point((quad_pts[2][1],quad_pts[2][2])), Point((quad_pts[3][1],quad_pts[3][2]))]
  elseif p == QUAD
    return [Point(quad_pts[1][1],quad_pts[1][2]), Point((quad_pts[2][1],quad_pts[2][2])), Point((quad_pts[3][1],quad_pts[3][2])), Point((quad_pts[4][1],quad_pts[4][2]))]
  else
    return quad_pts
  end
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

function _generate_face_nodes(nnodes,face_to_own_nodes,polytope,reffaces)

  face_to_num_fnodes = map(num_nodes,reffaces)
  push!(face_to_num_fnodes,nnodes)

  face_to_lface_to_own_fnodes = map(get_face_own_nodes,reffaces)
  push!(face_to_lface_to_own_fnodes,face_to_own_nodes)

  face_to_lface_to_face = get_faces(polytope)

_generate_face_nodes_aux(
  nnodes,
  face_to_own_nodes,
  face_to_num_fnodes,
  face_to_lface_to_own_fnodes,
  face_to_lface_to_face)
end

function _generate_face_dofs(ndofs,face_to_own_dofs,polytope,reffaces)

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

function _generate_face_nodes_aux(
nnodes,
face_to_own_nodes,
face_to_num_fnodes,
face_to_lface_to_own_fnodes,
face_to_lface_to_face)

if nnodes == length(face_to_own_nodes[end])
  face_fnode_to_node = fill(Int[],length(face_to_own_nodes))
  face_fnode_to_node[end] = collect(1:nnodes)
  return face_fnode_to_node
end

face_fnode_to_node = Vector{Int}[]
for (face, nfnodes) in enumerate(face_to_num_fnodes)
  fnode_to_node = zeros(Int,nfnodes)
  lface_to_face = face_to_lface_to_face[face]
  lface_to_own_fnodes = face_to_lface_to_own_fnodes[face]
  for (lface, faceto) in enumerate(lface_to_face)
    own_nodes = face_to_own_nodes[faceto]
    own_fnodes = lface_to_own_fnodes[lface]
    fnode_to_node[own_fnodes] = own_nodes
  end
  push!(face_fnode_to_node,fnode_to_node)
end

face_fnode_to_node
end

function _generate_face_own_dofs(face_own_nodes, node_and_comp_to_dof)
  faces = 1:length(face_own_nodes)
  T = eltype(node_and_comp_to_dof)
  comps = 1:num_components(T)
  face_own_dofs = [Int[] for i in faces]
  for face in faces
    nodes = face_own_nodes[face]
    # Node major
    for comp in comps
      for node in nodes
        comp_to_dofs = node_and_comp_to_dof[node]
        dof = comp_to_dofs[comp]
        push!(face_own_dofs[face],dof)
      end
    end
  end

  face_own_dofs

end

function compute_lagrangian_quad_reffaces(::Type{T},p::Polytope,orders,trian) where T
  _compute_lagrangian_quad_reffaces(T,p,orders,trian)
end

_compute_lagrangian_quad_reffaces(::Type{T},p::Polytope{0},orders,trian) where T = ()

function _compute_lagrangian_quad_reffaces(::Type{T},p::Polytope{D},orders,trian) where {T,D}
  reffaces = [ LagrangianRefFE{d}[]  for d in 0:D ]
  p0 = Polytope{0}(p,1)
  reffe0 = LagrangianQuadRefFE(T,p0,(),trian)
  for vertex in 1:num_vertices(p)
    push!(reffaces[0+1],reffe0)
  end
  offsets = get_offsets(p)
  for d in 1:(num_dims(p)-1)
    offset = offsets[d+1]
    for iface in 1:num_faces(p,d)
      face = Polytope{d}(p,iface)
      face_orders = compute_face_orders(p,face,iface,orders)
      refface = LagrangianQuadRefFE(T,face,face_orders,trian)
      push!(reffaces[d+1],refface)
    end
  end
  tuple(reffaces...)
end

function tfill(v, ::Val{D}) where D
  t = tfill(v, Val{D-1}())
  (v,t...)
end

tfill(v,::Val{0}) = ()
tfill(v,::Val{1}) = (v,)

tfill(v,::Val{2}) = (v,v)
tfill(v,::Val{3}) = (v,v,v)

function _default_space(p)
  if is_n_cube(p)
    :Q
  else
    :P
  end
end
