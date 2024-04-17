abstract type TensorProductFactors{T,N} <: AbstractArray{T,N} end

struct BasisFactors{T,A,I} <: TensorProductFactors{T,2}
  factors::A
  index_map::I
  function BasisFactors(
    factors::A,index_map::I,
    ) where {T,A<:AbstractVector{<:AbstractMatrix{T}},I<:NodesAndComps2DofsMap}
    new{T,A,I}(factors,index_map)
  end
end

Base.size(a::BasisFactors) = (a.index_map.ndofs,a.index_map.ndofs)
Base.axes(a::BasisFactors) = (Base.OneTo(a.index_map.ndofs),Base.OneTo(a.index_map.ndofs))

get_factors(a::BasisFactors) = a.factors

function Base.getindex(a::BasisFactors,i::Integer,j::Integer)
  factors = a.factors
  nnodes = num_nodes(a.index_map)
  ncomps = num_components(a.index_map)
  compi = FEM.slow_index(i,nnodes)
  compj = FEM.fast_index(j,ncomps)
  if compi != compj
    return zero(eltype(a))
  end
  nodei = FEM.fast_index(i,nnodes)
  nodej = FEM.slow_index(j,ncomps)
  rowi = a.index_map.nodes_map[nodei]
  colj = a.index_map.dofs_map[nodej]
  return prod(map(d->factors[d][rowi[d],colj[d]],eachindex(factors)))
end
