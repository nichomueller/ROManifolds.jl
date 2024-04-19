function _get_terms(orders::NTuple{D,Int}) where D
  return collect1d(CartesianIndices(orders.+1))
end

function _get_dof_to_node(::Type{T},nodes) where T
  nnodes = length(nodes)
  ncomps = num_components(T)
  repeat(Base.OneTo(nnodes),ncomps)
end

function _get_dof_to_comp(::Type{T},nodes) where T
  nnodes = length(nodes)
  ncomps = num_components(T)
  repeat(Base.OneTo(ncomps),inner=(nnodes,))
end

function _get_node_and_comp_to_dof(::Type{T},nodes) where T<:MultiValue
  nnodes = length(nodes)
  ncomps = num_components(T)
  dof_to_node = _get_dof_to_node(T,nodes)
  dof_to_comp = _get_dof_to_comp(T,nodes)
  ids = (dof_to_comp .- 1)*nnodes .+ dof_to_node
  rids = reshape(ids,nnodes,ncomps)
  [Point(rids[i,:]) for i = axes(rids,1)]
end

function _get_node_and_comp_to_dof(::Type{T},nodes) where T
  _get_dof_to_node(T,nodes)
end

ReferenceFEs.num_dofs(a::LagrangianDofBasis{P,V}) where {P,V} = length(a.nodes)*num_components(V)

function _split_cartesian_descriptor(origin::Point{D},sizes,partition,cmap,isperiodic) where D
  function _compute_1d_desc(
    o=first(origin.data),s=first(sizes),p=first(partition),m=cmap,i=first(isperiodic))
    CartesianDescriptor(Point(o),(s,),(p,);map=m,isperiodic=(i,))
  end
  isotropy = Isotropy(map(Isotropy,(sizes,partition,cmap,isperiodic))...)
  factors = isotropy==Isotropic() ? Fill(_compute_1d_desc(),D) : map(_compute_1d_desc,origin.data,sizes,partition,Fill(map,D),isperiodic)
  return factors,isotropy
end
