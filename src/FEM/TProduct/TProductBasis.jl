Base.promote_rule(::Type{VectorValue{1,T}},::Val{D}) where {D,T} = VectorValue{D,T}

struct TensorProductDofBases{D,T,A,B} <: AbstractVector{T}
  dof_bases::NTuple{D,A}
  nodes::B

  function TensorProductDofBases(
    dof_bases::NTuple{D,A},
    nodes::TensorProductNodes{D}) where {D,T1<:Dof,A<:AbstractVector{T1}}

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
