D = 2
reffe = ReferenceFE(QUAD,FEM.tplagrangian,VectorValue{2,Float64},2)

tpreffe = ReferenceFE(QUAD,lagrangian,VectorValue{2,Float64},(2,3))

tpreffe = reffe.tpreffe
dofs = tpreffe.reffe.dofs

reffex = reffe.reffe[1]
dofsx = reffex.reffe.dofs

monb = get_prebasis(tpreffe)
monbx = get_prebasis(reffex)

shapes = get_shapefuns(tpreffe)

x = get_data(get_cell_points(Ω))[1]
Mx = evaluate(monb,x)

function _index_inclusion(orders,α,β)
  N = num_vertices(SEGMENT) # 2
  if α ∈ β
    return N+1:orders[α]+1
  else
    return 1:N
  end
end

function push_entries!(vids,I::CartesianIndices{D},perm=1:D) where D
  ids = permutedims(collect(I),perm)
  for idi in ids
    push!(vids,idi)
  end
end

function indices_map(p::Polytope{D},orders::NTuple{D,Int}) where D
  vids = CartesianIndex{D}[]
  for d = 0:D
    I = collect(IterTools.subsets(1:D,Val(d)))
    for i in I
      ij = CartesianIndices(ntuple(j->_index_inclusion(orders,j,i),D))
      _push_entries!(vids,ij)
    end
  end
  return vids
end

struct TensorProductNodes{D,T} <: AbstractVector{Point{D,T}}
  nodes::NTuple{D,Point{1,T}}
  nodes_map::Vector{CartesianIndex{D}}

  function TensorProductNodes(
    nodes::NTuple{D,Point{1,T}},
    nodes_map::Vector{CartesianIndex{D}}) where {D,T<:Dof}

    A = typeof(nodes)
    B = typeof(nodes_map)
    new{T,D,A,B}(nodes,nodes_map)
  end
end

Base.length(a::TensorProductNodes) = length(a.nodes_map)
Base.size(a::TensorProductNodes) = (length(a),)
Base.axes(a::TensorProductNodes) = (Base.OneTo(length(a)),)
Base.IndexStyle(::TensorProductNodes) = IndexLinear()

function Base.getindex(a::TensorProductNodes{D,T},i::Integer) where {D,T}
  entry = a.nodes_map[i]
  p = zero(Mutable(Point{D,T}))
  @inbounds for d in 1:D
    p[d] = a.nodes[d][entry[d]]
  end
  Point(p)
end

Base.promote_rule(::Type{VectorValue{1,T}},::Val{D}) where D = VectorValue{D,T}

struct TensorProductMonomialBases{D,T} <: AbstractVector{Monomial}
  bases::NTuple{D,MonomialBasis{1,T}}
  basis::MonomialBasis{D,T}
end

function TensorProductMonomialBases(bases::NTuple{D,MonomialBasis{1,T}}) where {D,T}
  get_inner_terms(b::MonomialBasis) = b.terms
  _terms = get_inner_terms.(bases)
  nterms = map(length,_terms)
  terms = collect1d(CartesianIndices(nterms))
  orders = ntuple(i->get_order(bases[i]),D)
  basis = MonomialBasis{D}(T,orders,terms)
  TensorProductMonomialBases(bases,basis)
end

Base.size(a::TensorProductMonomialBases{D,T}) where {D,T} = size(a.basis)
Base.getindex(a::TensorProductMonomialBases,i::Integer) = Monomial()
Base.IndexStyle(::TensorProductMonomialBases) = IndexLinear()

ReferenceFEs.get_order(a::TensorProductMonomialBases) = get_order(a.basis)
ReferenceFEs.get_orders(a::TensorProductMonomialBases) = get_orders(a.basis)
ReferenceFEs.get_exponents(a::TensorProductMonomialBases) = get_exponents(a.basis)

function Arrays.return_cache(f::TensorProductMonomialBases{D,T},x::AbstractVector{<:Point}) where {D,T}
  return_cache(f.basis,x)
end

function Arrays.evaluate!(cache,f::TensorProductMonomialBases{D,T},x::AbstractVector{<:Point}) where {D,T}
  evaluate!(cache,f.basis,x)
end

function Arrays.return_cache(
  fg::FieldGradientArray{Ng,TensorProductMonomialBases{D,T}},
  x::AbstractVector{<:Point}) where {Ng,D,T}

  return_cache(FieldGradientArray{Ng}(fg.fa.basis),x)
end

function Arrays.evaluate!(
  cache,
  fg::FieldGradientArray{Ng,TensorProductMonomialBases{D,T}},
  x::AbstractVector{<:Point}) where {Ng,D,T}

  evaluate!(cache,FieldGradientArray{Ng}(fg.fa.basis),x)
end

struct TensorProductDofBases{D,T,A,B} <: AbstractVector{T}
  dof_bases::NTuple{D,A}
  nodes::B

  function TensorProductDofBases(
    dof_bases::NTuple{D,A},
    nodes::TensorProductNodes{D}) where {D,T1<:Dof,A<:AbstractVector{T}}

    T = promote_rule(T1,Val(D))
    B = typeof(nodes)
    new{D,T,A,B}(dof_bases,nodes)
  end
end

Base.length(a::TensorProductDofBases) = length(a.nodes)
Base.size(a::TensorProductDofBases) = (length(a),)
Base.axes(a::TensorProductDofBases) = (Base.OneTo(length(a)),)
Base.IndexStyle(::TensorProductDofBases) = IndexLinear()
Base.getindex(a::TensorProductDofBases,i::Integer) = PointValue(a.nodes[i])

ReferenceFEs.get_nodes(a::TensorProductDofBases) = a.nodes

TensorValues.num_components(::TensorProductDofBases{D,T,LagrangianDofBasis{T1,V}}) where {D,T,T1,V} = num_components(V)

function ReferenceFEs.get_dof_to_node(a::TensorProductDofBases{D,T,<:LagrangianDofBasis}) where {D,T}
  nnodes = length(a.nodes)
  ncomps = num_components(a)
  ids = ntuple(i->collect(Base.OneTo(nnodes)),ncomps)
  collect1d(ids)
end

function ReferenceFEs.get_dof_to_comp(a::TensorProductDofBases{D,T,<:LagrangianDofBasis}) where {D,T}
  nnodes = length(a.nodes)
  ncomps = num_components(a)
  ids = ntuple(i->fill(i,nnodes),ncomps)
  collect1d(ids)
end

function ReferenceFEs.get_node_and_comp_to_dof(a::TensorProductDofBases{D,T,<:LagrangianDofBasis}) where {D,T}
  nnodes = length(a.nodes)
  ncomps = num_components(a)
  dof_to_node = get_dof_to_node(a)
  dof_to_comp = get_dof_to_comp(a)
  ids = (dof_to_comp .- 1)*ncomps .+ dof_to_node
  rids = reshape(ids,nnodes,ncomps)
  [VectorValue[rids[i,:]] for i = axes(rids,1)]
end

function Arrays.return_cache(a::TensorProductDofBases{D,T,<:LagrangianDofBasis},field) where {D,T}
  dof_to_node = get_dof_to_node(a)
  node_and_comp_to_dof = get_node_and_comp_to_dof(a)
  cf = return_cache(field,a.nodes)
  vals = evaluate!(cf,field,a.nodes)
  ndofs = length(dof_to_node)
  r = ReferenceFEs._lagr_dof_cache(vals,ndofs)
  c = CachedArray(r)
  (c,cf,dof_to_node,node_and_comp_to_dof)
end

function Arrays.evaluate!(cache,a::TensorProductDofBases{D,T,<:LagrangianDofBasis},field) where {D,T}
  c,cf,dof_to_node,node_and_comp_to_dof = cache
  vals = evaluate!(cf,field,a.nodes)
  ndofs = length(dof_to_node)
  S = eltype(vals)
  ncomps = num_components(S)
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

function Arrays.return_cache(
  a::TensorProductDofBases{D,T,<:LagrangianDofBasis},
  field::TensorProductField) where {D,T}

  b = first(a.dof_bases)
  f = first(field.field)
  (c,cf,dof_to_node,node_and_comp_to_dof) = return_cache(b,f)
  vc = Vector{typeof(get_array(c))}(undef,D)
  C = _tp_lagr_dof_cache(c,length(get_dof_to_node(a)))
  (C,vc,c,cf,dof_to_node,node_and_comp_to_dof)
end

function Arrays.evaluate!(
  cache,
  a::TensorProductDofBases{D,T,<:LagrangianDofBasis},
  field::TensorProductField) where {D,T}

  C,vc,_cache... = cache
  @inbounds for d = 1:D
    vc[d] = evaluate!(_cache,a.dof_bases[d],field.field[d])
  end
  copyto!(C,kronecker(vc...))
  return C
end

function Arrays.return_cache(a::TensorProductDofBases{D,T,<:MomentBasedDofBasis},field) where {D,T}
  @notimplemented
end

function Arrays.evaluate!(cache,a::TensorProductDofBases{D,T,<:MomentBasedDofBasis},field) where {D,T}
  @notimplemented
end

struct TensorProductLinComb{D,V,F} <: AbstractVector{LinearCombinationField{V,F}}
  tplincomb::LinearCombinationFieldVector{V,F}
  lincombs::NTuple{D,LinearCombinationField{V,F}}
end

Base.size(a::TensorProductLinComb) = size(a.tplincomb)
Base.getindex(a::TensorProductLinComb,i::Integer) = getindex(a.tplincomb,i)
Base.IndexStyle(::Type{<:TensorProductLinComb}) = IndexLinear()

function ReferenceFEs.compute_shapefuns(
  dofs::TensorProductDofBases{D},
  prebasis::TensorProductMonomialBases{D}) where D

  changes = ntuple(i->inv(evaluate(dofs.dof_bases[i],prebasis.bases[i])),D)
  lincombs = ntuple(i->linear_combination(changes[i],prebasis.bases[i]),D)

  change = kronecker(changes...)
  tplincomb = linear_combination(change,prebasis.basis)

  TensorProductLinComb(tplincomb,lincombs)
end

struct TensorProductRefFE{D,A,B} <: ReferenceFE{D}
  reffe::NTuple{D,A}
  dof_basis::B

  function TensorProductRefFE(
    reffe::NTuple{D,A},
    dof_basis::TensorProductDofBases{D}
    ) where {A<:ReferenceFE{1}}

    B = typeof(dof_basis)
    new{D,A,B}(reffe,dof_basis)
  end
end

abstract type TensorProductRefFEName <: ReferenceFEName end

ureffe(::TensorProductRefFEName) = @abstractmethod

struct TensorProdLagrangian <: TensorProductRefFEName end

ureffe(::TensorProdLagrangian) = Lagrangian()

const tplagrangian = TensorProdLagrangian()

function ReferenceFEs.ReferenceFE(p::Polytope,name::TensorProductRefFEName,order)
  TensorProductRefFE(p,name,Float64,order)
end

function ReferenceFEs.ReferenceFE(p::Polytope,name::TensorProductRefFEName,::Type{T},order) where T
  TensorProductRefFE(p,name,T,order)
end

function TensorProductRefFE(p::Polytope{D},name::TensorProductRefFEName,::Type{T},order::Int) where {D,T}
  TensorProductRefFE(p,name,T,ntuple(i->order,D))
end

function TensorProductRefFE(p::Polytope{D},name::TensorProductRefFEName,::Type{T},orders) where {D,T}
  @check length(orders) == D
  nodes_map = indices_map(p,orders)

  reffes = ntuple(i->ReferenceFE(SEGMENT,ureffe(name),T,orders[i]),D)
  prebasis = TensorProductMonomialBases(get_prebasis.(reffes))
  nodes = TensorProductNodes(get_nodes.(reffes),nodes_map)
  dof_basis = TensorProductDofBases(get_dof_basis.(reffes),nodes)
  shapefuns = compute_shapefuns(dof_basis,prebasis)

  return TensorProductRefFE(reffes,dof_basis)
end

ReferenceFEs.num_dofs(reffe::TensorProductRefFE)      = prod(num_dofs.(reffe.reffe))
ReferenceFEs.get_polytope(reffe::TensorProductRefFE)  = Polytope(ntuple(i->HEX_AXIS,D)...)
ReferenceFEs.get_prebasis(reffe::TensorProductRefFE)  = reffe.prebasis
ReferenceFEs.get_dof_basis(reffe::TensorProductRefFE) = reffe.dof_basis
ReferenceFEs.Conformity(reffe::TensorProductRefFE)    = Conformity(first(reffe.reffe))
ReferenceFEs.get_face_dofs(reffe::TensorProductRefFE) = reffe.face_dofs
ReferenceFEs.get_shapefuns(reffe::TensorProductRefFE) = reffe.shapefuns
ReferenceFEs.get_metadata(reffe::TensorProductRefFE)  = reffe.metadata
ReferenceFEs.get_orders(reffe::TensorProductRefFE)    = ntuple(i->get_order(reffe.reffe[i]),D)
ReferenceFEs.get_order(reffe::TensorProductRefFE)     = maximum(get_orders(reffe))

ReferenceFEs.Conformity(reffe::TensorProductRefFE,sym::Symbol) = Conformity(reffe.reffe,sym)
ReferenceFEs.get_face_own_dofs(reffe::TensorProductRefFE,conf::Conformity) = get_face_own_dofs(reffe.reffe,conf)
