abstract type TensorProductFactors{T,N} <: AbstractArray{T,N} end

struct FieldFactors{I,T,A,B} <: TensorProductFactors{T,1}
  factors::A
  index_map::B
  isotropy::I
  function FieldFactors(
    factors::A,index_map::B,isotropy::I
    ) where {T,A<:AbstractVector{<:AbstractVector{T}},B<:NodesMap,I}
    new{I,T,A,B}(factors,index_map,isotropy)
  end
end

Base.size(a::FieldFactors) = (num_nodes(a.index_map),)
Base.axes(a::FieldFactors) = (Base.OneTo(num_nodes(a.index_map)),)

get_factors(a::FieldFactors) = a.factors

function Base.getindex(a::FieldFactors,i::Integer)
  factors = a.factors
  entry = a.nodes_map.indices[i]
  return prod(map(d->factors[d][entry[d]],eachindex(factors)))
end

struct BasisFactors{I,T,A,B} <: TensorProductFactors{T,2}
  factors::A
  index_map::B
  isotropy::I
  function BasisFactors(
    factors::A,index_map::B,isotropy::I
    ) where {T,A<:AbstractVector{<:AbstractMatrix{T}},B<:NodesAndComps2DofsMap,I}
    new{I,T,A,B}(factors,index_map,isotropy)
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
