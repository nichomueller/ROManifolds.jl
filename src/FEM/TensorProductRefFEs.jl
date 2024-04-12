struct TensorProductRefFE{D,R} <: ReferenceFE{D}
  reffe::R
  function TensorProductRefFE(reffe::NTuple{D,ReferenceFE{1}}) where D
    R = typeof(reffe)
    new{D,R}(reffe)
  end
end

struct TensorProdLagrangian <: ReferenceFEName end

const tplagrangian = TensorProdLagrangian()

function ReferenceFEs.ReferenceFE(
  polytope::NTuple{D,Polytope},
  ::TensorProdLagrangian,
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}};
  kwargs...) where {T,D}

  reffes = ntuple(i->ReferenceFE(polytope[i],lagrangian,T,orders;kwargs...),D)
  TensorProductRefFE(reffes)
end

struct TensorProduct1DVectorValue{T,D,V} <: AbstractArray{VectorValue{T,D},D}
  vv1d::V
  function TensorProduct1DVectorValue(vv1d::NTuple{D,AbstractVector{VectorValue{T,1}}}) where {T,D}
    V = typeof(vv1d)
    new{T,D,V}(vv1d)
  end
end

Base.size(a::TensorProduct1DVectorValue{T,D}) where {T,D} = length.(a.vv1d)
Base.axes(a::TensorProduct1DVectorValue{T,D}) where {T,D} = Base.OneTo(size.(a))
Base.IndexStyle(::TensorProduct1DVectorValue) = IndexCartesian()

function Base.getindex(a::TensorProduct1DVectorValue{T,D},i::Vararg{Integer,D}) where {D,T}
  p = zero(Mutable(Point{D,T}))
  @inbounds for d in 1:D
    p[d] = a.vv1d[i[d]]
  end
  p
end

function tprod(t::NTuple{N,Type{T}}) where {N,T}
  v = map(testargs,t)
  typeof(tprod(v))
end

function tprod(v::NTuple{N,<:Number}) where N
  collect1d(v)
end

function tprod(v::NTuple{N,VectorValue{T,1}}) where {N,T}
  collect1d(v)
end

function tprod(v::NTuple{N,<:AbstractVector{<:Number}}) where N
  kronecker(v...)
end

function tprod(v::NTuple{N,<:AbstractVector{VectorValue{T,1}}}) where {N,T}
  TensorProduct1DVectorValue(v)
end

struct TensorProductLagrangianDofBasis{P,V,D,B} <: AbstractVector{ReferenceFEs.PointValue{P}}
  bases::NTuple{D,B}
  function TensorProductLagrangianDofBasis(bases::NTuple{D,B}) where {D,P1,V1,B<:LagrangianDofBasis{P1,V1}}
    P = tprod(ntuple(i->P1,D))
    V = tprod(ntuple(i->V1,D))
    new{P,V,D,B}(bases)
  end
end

get_nnodes(a::TensorProductLagrangianDofBasis) = length(get_nodes(a))
get_ncomps(a::TensorProductLagrangianDofBasis{P,V}) where {P,V} = TensorValues.num_components(V)
get_ndofs(a::TensorProductLagrangianDofBasis) = get_nnodes(a)*get_ncomps(a)

Base.size(a::TensorProductLagrangianDofBasis) = (get_nnodes(a),)
Base.axes(a::TensorProductLagrangianDofBasis) = (axes(get_nodes(a),1),)
Base.getindex(a::TensorProductLagrangianDofBasis,i::Integer) = PointValue(get_nodes(a)[i])
Base.IndexStyle(::TensorProductLagrangianDofBasis) = IndexLinear()

function ReferenceFEs.get_nodes(a::TensorProductLagrangianDofBasis)
  tprod_nodes = tprod(get_nodes.(a))
  collect1d(tprod_nodes)
end

function ReferenceFEs.get_dof_to_node(a::TensorProductLagrangianDofBasis)
  nnodes = get_nnodes(a)
  ncomps = get_ncomps(a)
  ids = ntuple(i->collect(Base.OneTo(nnodes)),ncomps)
  collect1d(ids)
end

function ReferenceFEs.get_dof_to_comp(a::TensorProductLagrangianDofBasis)
  nnodes = get_nnodes(a)
  ncomps = get_ncomps(a)
  ids = ntuple(i->fill(i,nnodes),ncomps)
  collect1d(ids)
end

function get_node_and_comp_to_dof(a::TensorProductLagrangianDofBasis{P,V}) where {P,V}
  nnodes = get_nnodes(a)
  ncomps = get_ncomps(a)
  dof_to_node = get_dof_to_node(a)
  dof_to_comp = get_dof_to_comp(a)
  ids = (dof_to_comp .- 1)*ncomps .+ dof_to_node
  rids = reshape(ids,nnodes,ncomps)
  [VectorValue[rids[i,:]] for i = axes(rids,1)]
end

function Arrays.return_cache(b::TensorProductLagrangianDofBasis,field)
  nodes = get_nodes(b)
  dof_to_node = get_dof_to_node(b)
  cf = return_cache(field,nodes)
  vals = evaluate!(cf,field,nodes)
  ndofs = length(dof_to_node)
  r = ReferenceFEs._lagr_dof_cache(vals,ndofs)
  c = CachedArray(r)
  (c, cf)
end

function Arrays.evaluate!(cache,b::TensorProductLagrangianDofBasis,field)
  c,cf = cache
  nodes = get_nodes(b)
  dof_to_node = get_dof_to_node(b)
  node_and_comp_to_dof = get_node_and_comp_to_dof(b)
  vals = evaluate!(cf,field,nodes)
  ndofs = length(dof_to_node)
  T = eltype(vals)
  ncomps = num_components(T)
  @check ncomps == num_components(eltype(node_and_comp_to_dof)) """\n
  Unable to evaluate LagrangianDofBasis. The number of components of the
  given Field does not match with the LagrangianDofBasis.

  If you are trying to interpolate a function on a FESpace make sure that
  both objects have the same value type.

  For instance, trying to interpolate a vector-valued function on a scalar-valued FE space
  would raise this error.
  """
  ReferenceFEs._evaluate_lagr_dof!(c,vals,node_and_comp_to_dof,ndofs,ncomps)
end

function get_terms(terms::AbstractVector{CartesianIndex{1}}...)
  nterms = map(length,terms)
  collect1d(CartesianIndices(nterms))
end

function get_terms(r::TensorProductRefFE)
  _get_terms(b::MonomialBasis) = b.terms
  prebases = get_prebasis.(r.reffe)
  get_terms(_get_terms.(prebases)...)
end

function conformity_check(c::Vararg{Conformity})
  conformity = first(c)
  @check all(c .== conformity) "Tensor product refFEs must have same conformity trait"
  return conformity
end

function ReferenceFEs.get_face_own_nodes(r::TensorProductRefFE{D}) where D
  p = get_polytope(r)
  orders = get_orders(r)
  _get_face_own_nodes(p,orders)
end

function _get_face_own_nodes(p::Polytope,orders::NTuple)
  if any(map(i->i==0,orders))
    _compute_constant_face_own_nodes(p,orders)
  elseif all(map(i->i==1,orders))
    _compute_linear_face_own_nodes(p)
  else
    _compute_high_order_face_own_nodes(p,orders)
  end
end

function _compute_constant_face_own_nodes(p,orders)
  @assert all( orders .== 0) "If an order is 0 in some direction, it should be 0 also in the others"
  facenodes = [Int[] for i in 1:num_faces(p)]
  push!(facenodes[end],1)
  facenodes
end

function _compute_linear_face_own_nodes(p)
  facenodes = [Int[] for i in 1:num_faces(p)]
  for i in 1:num_vertices(p)
    push!(facenodes[i],i)
  end
  facenodes
end

function _compute_high_order_face_own_nodes(p::Polytope{D},orders) where D
  facenodes = [Int[] for i in 1:num_faces(p)]
  _compute_high_order_face_own_nodes_dim_0!(facenodes,p)
  for d in 1:(num_dims(p)-1)
    _compute_high_order_face_own_nodes_dim_d!(facenodes,p,orders,Val{d}())
  end
  _compute_high_order_face_own_nodes_dim_D!(facenodes,p,orders)
  facenodes
end

function _compute_high_order_face_own_nodes_dim_0!(facenodes,p)
  k = 1
  for vertex in 1:num_vertices(p)
    push!(facenodes[vertex],k)
    k += 1
  end
end

@noinline function _compute_high_order_face_own_nodes_dim_d!(facenodes,p,orders,::Val{d}) where d
  offset = get_offset(p,d)
  k = length(nodes)+1
  for iface in 1:num_faces(p,d)
    face_orders = compute_face_orders(p,face,iface,orders)
    for _ in 1:prod(face_orders)
      push!(facenodes[iface+offset],k)
      k += 1
    end
  end
end

function _compute_high_order_face_own_nodes_dim_D!(facenodes,p::ExtrusionPolytope{D},orders) where D
  k = length(facenodes)+1
  p_inner_x = compute_own_nodes(p,orders)
  for _ in 1:length(p_inner_x)
    push!(facenodes[end],k)
    k += 1
  end
end

get_type(prebasis::MonomialBasis{D,T}) where {D,T} = T
get_type(r::TensorProductRefFE) = get_type(get_prebasis(r.reffe[1]))

ReferenceFEs.get_order(r::TensorProductRefFE) = minimum(get_orders(r))

ReferenceFEs.get_orders(r::TensorProductRefFE{D}) where D = ntuple(i->get_order(r.reffe[i]),D)

ReferenceFEs.num_dofs(r::TensorProductRefFE) = prod(num_dofs.(r.reffe))

ReferenceFEs.get_polytope(r::TensorProductRefFE{D}) where D = Polytope(ntuple(i->HEX_AXIS,D)...)

ReferenceFEs.get_prebasis(r::TensorProductRefFE{D}) where D = MonomialBasis{D}(get_type(r),get_orders(r),get_terms(r))

ReferenceFEs.get_dof_basis(r::TensorProductRefFE) = TensorProductLagrangianDofBasis(get_dof_basis.(r))

ReferenceFEs.Conformity(r::TensorProductRefFE) = conformity_check(Conformity.(r.reffe))

function ReferenceFEs.get_shapefuns(r::TensorProductRefFE)
  _dofs = get_separate_dof_basis(r)
  _prebasis = get_separate_prebasis(r)
  prebasis = get_prebasis(r)
  changes = inv.(evaluate.(_dofs,_prebasis))
  change = kronecker(changes...)
  compute_shapefuns(change,prebasis)
end

function ReferenceFEs.get_face_dofs(r::TensorProductRefFE)
  polytope = get_polytope(r)
  orders = get_orders(r)
  dofs = get_dofs(r)
  T = get_type(r)
  if any(map(i->i==0,orders)) && !all(map(i->i==0,orders))
    cont = map(i -> i == 0 ? DISC : CONT,orders)
    CDTensorProductFaceDofs(T,polytope,orders,dofs,cont)
  else
    TensorProductFaceDofs(T,polytope,orders,dofs)
  end
end

struct TensorProductFaceDofs{T,D,P,B} <: AbstractVector{Vector{Int}}
  polytope::P
  orders::NTuple{D,Int}
  dofs::B
  function TensorProductFaceDofs(
    ::Type{T},
    polytope::Polytope{D},
    orders::NTuple{D,Int},
    dofs::LagrangianDofBasis) where {T,D}

    P = typeof(polytope)
    B = typeof(dofs)
    new{T,D,P,B}(polytope,orders,dofs)
  end
end

Base.length(a::TensorProductFaceDofs) = prod(a.orders .+ 1)
Base.size(a::TensorProductFaceDofs) = (length(a),)
Base.IndexStyle(::TensorProductFaceDofs) = IndexLinear()

function Base.getindex(a::TensorProductFaceDofs{T,D},iface::Integer) where {T,D}
  dim = D
  for d = 0:D-1
    if i <= get_offset(a.polytope,d)
      dim = d
      break
    end
  end
  face = Polytope{d}(a.polytope,iface)
  if d == 0
    p0 = Polytope{0}(a.polytope,1)
    refface = LagrangianRefFE(T,p0,())
  else
    face_orders = compute_face_orders(a.polytope,face,iface,a.orders)
    refface = LagrangianRefFE(T,face,face_orders)
  end
  if d == D
    nnodes = get_nnodes(a.dofs)
    ndofs = get_ndofs(a.dofs)
  else
    nnodes = num_nodes(refface)
    ndofs = num_dofs(refface)
  end

  face_own_nodes = _get_face_own_nodes(p,orders)[iface]
  face_own_dofs = _get_face_own_dofs(face_own_nodes,dofs.node_and_comp_to_dof)
  face_dofs = _get_face_dofs(ndofs,face_own_dofs,a.polytope,refface)

  return face_dofs
end

struct TensorProductFaceNodes{T,D,P,B} <: AbstractVector{Vector{Int}}
  polytope::P
  orders::NTuple{D,Int}
  dofs::B
  function TensorProductFaceNodes(
    ::Type{T},
    polytope::Polytope{D},
    orders::NTuple{D,Int},
    dofs::LagrangianDofBasis) where {T,D}

    P = typeof(polytope)
    B = typeof(dofs)
    new{T,D,P,B}(polytope,orders,dofs)
  end
end

Base.length(a::TensorProductFaceNodes) = prod(a.orders .+ 1)
Base.size(a::TensorProductFaceNodes) = (length(a),)
Base.IndexStyle(::TensorProductFaceNodes) = IndexLinear()

function Base.getindex(a::TensorProductFaceNodes{T,D},iface::Integer) where {T,D}
  dim = D
  for d = 0:D-1
    if i <= get_offset(a.polytope,d)
      dim = d
      break
    end
  end
  face = Polytope{d}(a.polytope,iface)
  if d == 0
    p0 = Polytope{0}(a.polytope,1)
    refface = LagrangianRefFE(T,p0,())
  else
    face_orders = compute_face_orders(a.polytope,face,iface,a.orders)
    refface = LagrangianRefFE(T,face,face_orders)
  end
  if d == D
    nnodes = get_nnodes(a.dofs)
    ndofs = get_ndofs(a.dofs)
  else
    nnodes = num_nodes(refface)
    ndofs = num_dofs(refface)
  end

  face_own_nodes = _get_face_own_nodes(p,orders)[iface]

  face_to_num_fnodes = num_nodes(refface)
  face_to_lface_to_own_fnodes = get_face_own_nodes(refface)
  face_to_lface_to_face = get_faces(a.polytope)[iface]
  if nnodes == length(face_own_nodes)
    return iface == length(a) ? collect(1:nnodes) : Int[]
  end

  face_nodes = zeros(Int,face_to_num_fnodes)
  lface_to_face = face_to_lface_to_face[iface]
  lface_to_own_fnodes = face_to_lface_to_own_fnodes[iface]
  for (lface,faceto) in enumerate(lface_to_face)
    own_nodes = face_to_own_nodes[faceto]
    own_fnodes = lface_to_own_fnodes[lface]
    face_nodes[own_fnodes] = own_nodes
  end
  return face_nodes
end

function _get_face_own_dofs(nodes,node_and_comp_to_dof)
  T = eltype(node_and_comp_to_dof)
  comps = 1:num_components(T)
  face_own_dofs = Int[]
  for comp in comps
    for node in nodes
      comp_to_dofs = node_and_comp_to_dof[node]
      dof = comp_to_dofs[comp]
      push!(face_own_dofs,dof)
    end
  end
  return face_own_dofs
end
