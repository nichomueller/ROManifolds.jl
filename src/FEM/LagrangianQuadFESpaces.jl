abstract type LagrangianQuadRefFE{D} <: Gridap.ReferenceFEs.LagrangianRefFE{D} end
struct LagrangianQuad <: Gridap.ReferenceFEs.ReferenceFEName end
const lagrangian_quad = LagrangianQuad()

function Gridap.ReferenceFEs.ReferenceFE(
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

function _lagrangian_quad_ref_fe(
  ::Type{T},
  p::Polytope{D},
  orders) where {T,D}

  #= @assert isa(p,ExtrusionPolytope)
  @assert is_n_cube(p)
  degrees = broadcast(*,2,orders)
  q = Quadrature(p,Gridap.ReferenceFEs.TensorProduct(),degrees) =#
  @assert isa(p,ExtrusionPolytope)
  @assert is_n_cube(p) || is_simplex(p) "Wrong polytope"
  q = Quadrature(p,2*last(orders))
  nodes = get_coordinates(q)

  prebasis = compute_monomial_basis(T,p,orders)

  # Compute face_own_nodes
  face_nodes = [Int[] for i in 1:num_faces(p)]
  push!(last(face_nodes),collect(1:length(nodes))...)

  # Compute face_own_dofs
  face_dofs = [Int[] for i in 1:num_faces(p)]
  push!(last(face_dofs),collect(1:length(nodes)*num_components(T))...)

  dofs = LagrangianDofBasis(T,nodes)

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
