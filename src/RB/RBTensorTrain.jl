# D is the dimension, N is the ndim of the solution vector. For scalar problems,
# N = 1 and thus we simply return the entry as if we had provided a linear index.
# For vector valued problems, N is equals to D and it returns a D-dimensional
# slice of the input array

struct Cartesian2Linear{N,D} <: Base.AbstractCartesianIndex
  index::CartesianIndex{N}
  sizes::NTuple{D,Integer}
end

Base.getindex(a::AbstractArray,i::Cartesian2Linear{1}) = getindex(a,i.I...)

function index_in_dimension(c2l::Cartesian2Linear{D,D},dim::Integer) where D
  @check c2l.index.I[dim] <= c2l.sizes[dim]
  return D*(c2l.index.I[dim]-1)+1
end

@generated function Base.getindex(a::AbstractArray,c2l::Cartesian2Linear{D,D}) where D
  s = "["
  for i in 1:D
    s *= "getindex($a,index_in_dimension($c2l,$i)), "
  end
  s *= "]"
  Meta.parse(s)
end

abstract type TTSnapshots{M,T} <: AbstractSnapshots{M,T} end

function Base.getindex(s::TTSnapshots,i...)

end
