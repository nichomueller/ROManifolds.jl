abstract type Contribution end

CellData.get_domains(a::Contribution) = a.trians
get_values(a::Contribution) = a.values

Base.length(a::Contribution) = length(a.values)
Base.size(a::Contribution,i...) = size(a.values,i...)
Base.getindex(a::Contribution,i...) = a.values[i...]
Base.setindex!(a::Contribution,v,i...) = a.values[i...] = v
Base.eachindex(a::Contribution) = eachindex(a.values)

@inline function contribution(f,trians)
  values = map(f,trians)
  @check typeof(trians) <: Tuple{Vararg{Triangulation}}  "$(typeof(trians))"
  Contribution(values,trians)
end

function contribution!(a,values)
  a.values .= values
end

function contribution!(a,f,trians)
  contribution!(a,map(f,trians))
end

function Base.getindex(a::Contribution,trian::Triangulation...)
  perm = FEM.find_permutation(trian,a.trians)
  getindex(a,perm...)
end

function Contribution(v::Tuple{Vararg{AbstractArray}},t::Tuple{Vararg{Triangulation}})
  ArrayContribution(v,t)
end

struct ArrayContribution{T,V,K} <: Contribution
  values::V
  trians::K
  function ArrayContribution(
    values::V,
    trians::K
    ) where {T,V<:Tuple{Vararg{AbstractArray{T}}},K<:Tuple{Vararg{Triangulation}}}

    @check length(values) == length(trians)
    @check !any([t === first(trians) for t = trians[2:end]])
    new{T,V,K}(values,trians)
  end
end

ArrayContribution(v::V,t::Triangulation) where V = ArrayContribution((v,),(t,))

Base.eltype(::ArrayContribution{T}) where T = T
Base.eltype(::Type{<:ArrayContribution{T}}) where T = T
Base.copy(a::ArrayContribution) = ArrayContribution(copy(a.values),a.trians)

struct ContributionBroadcast{D,T}
  contrib::D
  trians::T
end

function Base.broadcasted(f,a::ArrayContribution,b::Number)
  ContributionBroadcast(map(values -> Base.broadcasted(f,values,b),a.values),a.trians)
end

function Base.materialize(c::ContributionBroadcast)
  ArrayContribution(map(Base.materialize,c.contrib),c.trians)
end

function Base.materialize!(a::ArrayContribution,c::ContributionBroadcast)
  a.trians .= c.trians
  map(Base.materialize!,a.values,c.contrib.values)
  a.values = ArrayContribution(map(Base.materialize,c.contrib),c.trians)
  a
end

function LinearAlgebra.fillstored!(a::ArrayContribution,v)
  for vals in a.values
    LinearAlgebra.fillstored!(vals,v)
  end
  a
end

# quite hacky, it's to deal with multiple jacobians at the same time
Base.eltype(::Tuple{Vararg{ArrayContribution{T}}}) where T = T
Base.eltype(::Type{<:Tuple{Vararg{ArrayContribution{T}}}}) where T = T

function LinearAlgebra.fillstored!(a::Tuple{Vararg{A}},v) where {A<:ArrayContribution}
  for ai in a
    LinearAlgebra.fillstored!(ai,v)
  end
  a
end
