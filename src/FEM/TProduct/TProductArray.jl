abstract type TensorProductFactors{T,N} <: AbstractArray{T,N} end

struct FieldFactors{I,T,A,B} <: TensorProductFactors{T,1}
  factors::A
  indices_map::B
  isotropy::I
  function FieldFactors(
    factors::A,indices_map::B,isotropy::I
    ) where {T,A<:AbstractVector{<:AbstractVector{T}},B<:NodesMap,I}
    new{I,T,A,B}(factors,indices_map,isotropy)
  end
end

Base.size(a::FieldFactors) = (num_nodes(a.indices_map),)
Base.axes(a::FieldFactors) = (Base.OneTo(num_nodes(a.indices_map)),)

get_factors(a::FieldFactors) = a.factors
get_indices_map(a::FieldFactors) = a.indices_map

function Base.getindex(a::FieldFactors,i::Integer)
  factors = get_factors(a)
  entry = get_indices_map(a)[i]
  return prod(map(d->factors[d][entry[d]],eachindex(factors)))
end

struct BasisFactors{I,T,A,B} <: TensorProductFactors{T,2}
  factors::A
  indices_map::B
  isotropy::I
  function BasisFactors(
    factors::A,indices_map::B,isotropy::I
    ) where {T,A<:AbstractVector{<:AbstractMatrix{T}},B<:NodesAndComps2DofsMap,I}
    new{I,T,A,B}(factors,indices_map,isotropy)
  end
end

Base.size(a::BasisFactors) = (a.indices_map.ndofs,a.indices_map.ndofs)
Base.axes(a::BasisFactors) = (Base.OneTo(a.indices_map.ndofs),Base.OneTo(a.indices_map.ndofs))

get_factors(a::BasisFactors) = a.factors
get_indices_map(a::BasisFactors) = a.indices_map

function Base.getindex(a::BasisFactors,i::Integer,j::Integer)
  factors = get_factors(a)
  indices_map = get_indices_map(a)
  nnodes = num_nodes(indices_map)
  ncomps = num_components(indices_map)
  compi = FEM.slow_index(i,nnodes)
  compj = FEM.fast_index(j,ncomps)
  if compi != compj
    return zero(eltype(a))
  end
  nodei = FEM.fast_index(i,nnodes)
  nodej = FEM.slow_index(j,ncomps)
  rowi = indices_map.nodes_map[nodei]
  colj = indices_map.dofs_map[nodej]
  return prod(map(d->factors[d][rowi[d],colj[d]],eachindex(factors)))
end
