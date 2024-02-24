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
  Contribution(values,trians)
end

function contribution!(a,values)
  a.values .= values
end

function contribution!(a,f,trians)
  contribution!(a,map(f,trians))
end

Contribution(v::Vector{<:AbstractArray},t::Vector{<:Triangulation}) = ArrayContribution(v,t)

struct ArrayContribution{T,V,K} <: Contribution
  values::Vector{V}
  trians::Vector{K}
  function ArrayContribution(
    values::Vector{V},
    trians::Vector{K}
    ) where {T,V<:AbstractArray{T},K<:Triangulation}

    @check length(values) == length(trians)
    @check !any([t === first(trians) for t = trians[2:end]])
    new{T,V,K}(values,trians)
  end
end

ArrayContribution(v::V,t::Triangulation) where V = ArrayContribution([v],[t])

Base.eltype(::ArrayContribution{T}) where T = T
Base.eltype(::Type{<:ArrayContribution{T}}) where T = T
Base.copy(a::ArrayContribution) = ArrayContribution(copy(a.values),a.trians)

function Base.getindex(a::ArrayContribution,trian::Triangulation...)
  perm = FEM.find_permutation(a.trian,trian...)
  getindex(a,perm...)
end

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
