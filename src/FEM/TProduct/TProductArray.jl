mydot(a::NTuple{N,Number}) where N = mydot(a...)
mydot(a::Number...) = mydot(a[1],mydot(a[2:end]...))
mydot(a::Number,b::Number) = dot(a,b)
mydot(a::Number) = a
mydot(a::VectorValue,b::Number) = sum(a)*b
mydot(a::Number,b::VectorValue) = sum(b)*a
mydot(a::VectorValue,b::VectorValue) = dot(a,b)
mydot(a::VectorValue) = sum(a)

myprod(a::NTuple{N,Number}) where N = myprod(a...)
myprod(a::Number...) = myprod(a[1],myprod(a[2:end]...))
myprod(a::Number,b::Number) = *(a,b)
myprod(a::Number) = a
myprod(a::VectorValue,b::VectorValue) = Point(map(*,a.data,b.data))

abstract type TensorProductFactors{T,N} <: AbstractArray{T,N} end

struct FieldFactors{I,T,N,A,B} <: TensorProductFactors{T,N}
  factors::A
  indices_map::B
  isotropy::I
  function FieldFactors(
    factors::A,indices_map::B,isotropy::I=Isotropy(factors)
    ) where {T,N,A<:AbstractVector{<:AbstractArray{T,N}},B<:IndexMap,I}
    new{I,T,N,A,B}(factors,indices_map,isotropy)
  end
end

get_factors(a::FieldFactors) = a.factors
get_indices_map(a::FieldFactors) = a.indices_map

const VFieldFactors = FieldFactors{I,T,1,A,B} where {I,T,A,B}
const MFieldFactors = FieldFactors{I,T,2,A,B} where {I,T,A,B}

Base.size(a::VFieldFactors) = (num_nodes(a.indices_map),)
Base.axes(a::VFieldFactors) = (Base.OneTo(num_nodes(a.indices_map)),)

function Base.getindex(a::VFieldFactors,i::Integer)
  factors = get_factors(a)
  entry = get_indices_map(a)[i]
  return myprod(map(d->factors[d][entry[d]],eachindex(factors)))
end

Base.size(a::MFieldFactors) = (num_nodes(a.indices_map),num_dofs(a.indices_map))
Base.axes(a::MFieldFactors) = (Base.OneTo(num_nodes(a.indices_map)),Base.OneTo(num_dofs(a.indices_map)))

function Base.getindex(a::MFieldFactors,nodei::Integer,j::Integer)
  factors = get_factors(a)
  indices_map = get_indices_map(a)
  ncomps = num_components(indices_map)
  compj = FEM.fast_index(j,ncomps)
  nodej = FEM.slow_index(j,ncomps)
  rowi = indices_map.nodes_map[nodei,1]
  colj = indices_map.dofs_map[nodej,compj]
  return myprod(ntuple(d->factors[d][rowi[d],colj[d]],length(factors)))
end

# Vector-valued bases are flattened out. For e.g., a 1x2 matrix of eltype
# VectorValue{2,T} becomes a 2x2 matrix of eltype T

struct BasisFactors{I,T,A,B} <: TensorProductFactors{T,2}
  factors::A
  indices_map::B
  isotropy::I
  function BasisFactors(
    factors::A,indices_map::B,isotropy::I=Isotropy(factors)
    ) where {T,A<:AbstractVector{<:AbstractMatrix{T}},B<:IndexMap,I}
    E = eltype(T)
    new{I,E,A,B}(factors,indices_map,isotropy)
  end
end

Base.size(a::BasisFactors) = (num_dofs(a.indices_map),num_dofs(a.indices_map))
Base.axes(a::BasisFactors) = (Base.OneTo(num_dofs(a.indices_map)),Base.OneTo(num_dofs(a.indices_map)))

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
  rowi = indices_map.nodes_map[nodei,compi]
  colj = indices_map.dofs_map[nodej,compj]
  return mydot(ntuple(d->factors[d][rowi[d],colj[d]],length(factors)))
end

function compose(factors::TensorProductFactors,indices_map)
  CompositeTensorProductFactors(factors,indices_map)
end

struct CompositeTensorProductFactors{T,N,A,B} <: TensorProductFactors{T,N}
  factors::A
  indices_map::B
  function CompositeTensorProductFactors(
    factors::A,
    indices_map::B
    ) where {T,N,A<:TensorProductFactors{T,N},B}
    new{T,N,A,B}(factors,indices_map)
  end
end

get_factors(a::CompositeTensorProductFactors) = a.factors
get_indices_map(a::CompositeTensorProductFactors) = a.indices_map

Base.size(a::CompositeTensorProductFactors) = size(a.factors)
Base.axes(a::CompositeTensorProductFactors) = axes(a.factors)

# Base.IndexStyle(::CompositeTensorProductFactors) = IndexLinear()

function Base.getindex(a::CompositeTensorProductFactors,i::Integer...)
  factors = get_factors(a)
  mappedi = get_indices_map(a)[i...]
  getindex(factors,mappedi)
end
