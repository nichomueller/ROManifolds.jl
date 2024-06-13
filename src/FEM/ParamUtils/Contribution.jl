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
  perm = find_permutation(trian,a.trians)
  getindex(a,perm...)
end

Contribution(v::V,t::Triangulation) where V = Contribution((v,),(t,))

function Contribution(
  v::Tuple{Vararg{AbstractArray{T,N}}},
  t::Tuple{Vararg{Triangulation}}) where {T,N}

  ArrayContribution{T,N}(v,t)
end

function Contribution(
  v::Tuple{Vararg{ArrayBlock{T,N}}},
  t::Tuple{Vararg{Triangulation}}) where {T,N}

  ArrayContribution{T,N}(v,t)
end

struct ArrayContribution{T,N,V,K} <: Contribution
  values::V
  trians::K
  function ArrayContribution{T,N}(values::V,trians::K) where {T,N,V,K}
    @check length(values) == length(trians)
    @check !any([t === first(trians) for t = trians[2:end]])
    new{T,N,V,K}(values,trians)
  end
end

const VectorContribution{T,V,K} = ArrayContribution{T,1,V,K}
const MatrixContribution{T,V,K} = ArrayContribution{T,2,V,K}

Base.eltype(::ArrayContribution{T}) where T = T
Base.eltype(::Type{<:ArrayContribution{T}}) where T = T
Base.ndims(::ArrayContribution{T,N}) where {T,N} = N
Base.ndims(::Type{<:ArrayContribution{T,N}}) where {T,N} = N
Base.copy(a::ArrayContribution) = Contribution(copy(a.values),a.trians)

Base.sum(a::ArrayContribution) = sum(a.values)

function Base.fill!(a::ArrayContribution,v)
  for vals in a.values
    fill!(vals,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::ArrayContribution,v)
  for vals in a.values
    LinearAlgebra.fillstored!(vals,v)
  end
  a
end

function LinearAlgebra.mul!(
  c::VectorContribution,
  a::MatrixContribution,
  b::AbstractVector,
  α::Number,β::Number)

  for c in c.values, a in a.values
    mul!(c,a,b,α,β)
  end
  a
end

function LinearAlgebra.mul!(
  c::VectorContribution,
  a::MatrixContribution,
  b::VectorContribution,
  α::Number,β::Number)

  @check length(c) == length(b)
  for (c,b) in zip(c.values,b.values), a in a.values
    mul!(c,a,b,α,β)
  end
  a
end

function LinearAlgebra.axpy!(α::Number,a::ArrayContribution,b::ArrayContribution)
  @check length(a) == length(b)
  for (a,b) in (a.values,b.values)
    axpy!(α,a,b)
  end
  b
end

function Algebra.copy_entries!(a::ArrayContribution,b::ArrayContribution)
  @check length(a) == length(b)
  for (a,b) in zip(a.values,b.values)
    copy_entries!(a,b)
  end
  a
end

struct ContributionBroadcast{D,T}
  contrib::D
  trians::T
end

function Base.broadcasted(f,a::ArrayContribution,b::Number)
  ContributionBroadcast(map(values -> Base.broadcasted(f,values,b),a.values),a.trians)
end

function Base.materialize(c::ContributionBroadcast)
  Contribution(map(Base.materialize,c.contrib),c.trians)
end

function Base.materialize!(a::ArrayContribution,c::ContributionBroadcast)
  @check a.trians === c.trians
  map(Base.materialize!,a.values,c.contrib)
  a
end

# quite hacky

const TupOfArrayContribution = Tuple{Vararg{ArrayContribution{T}}} where T

Base.eltype(::TupOfArrayContribution{T}) where T = T
Base.eltype(::Type{<:TupOfArrayContribution{T}}) where T = T

function LinearAlgebra.fillstored!(a::TupOfArrayContribution,v)
  for ai in a
    LinearAlgebra.fillstored!(ai,v)
  end
  a
end

function Algebra.copy_entries!(a::TupOfArrayContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  for (a,b) in zip(a,b)
    copy_entries!(a,b)
  end
  a
end
