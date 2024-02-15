abstract type AbstractContribution end

Base.getindex(a::AbstractContribution,trian::Triangulation) = get_contribution(a,trian)
Base.setindex!(a::AbstractContribution,b,trian::Triangulation) = add_contribution!(a,b,trian)
CellData.num_domains(a::AbstractContribution) = length(a.dict)

struct Contribution{T} <: AbstractContribution
  dict::IdDict{Triangulation,T}
end

CellData.get_domains(a::Contribution) = collect(keys(a.dict))
get_values(a::Contribution) = collect(values(a.dict))

function CellData.get_contribution(a::T,trian::Triangulation) where T<:Contribution
  if haskey(a.dict,trian)
     return a.dict[trian]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this $T object.
    """
  end
end

function CellData.add_contribution!(a::Contribution{T},b::T,trian::Triangulation) where T
  if haskey(a.dict,trian)
    a.dict[trian] += b
  else
    a.dict[trian] = b
  end
  a
end

const ArrayContribution = Contribution{AbstractArray}

array_contribution() = Contribution(IdDict{Triangulation,AbstractArray}())

struct ContributionBroadcast{D}
  contrib::D
end

function Base.broadcasted(f,a::ArrayContribution,b::Number)
  c = Contribution(IdDict{Triangulation,ParamBroadcast}())
  for (trian,values) in a.dict
    c[trian] = Base.broadcasted(f,values,b)
  end
  ContributionBroadcast(c)
end

function Base.materialize(c::ContributionBroadcast)
  a = array_contribution()
  for (trian,b) in c.contrib.dict
    a[trian] = Base.materialize(b)
  end
  a
end

function Base.materialize!(a::ArrayContribution,c::ContributionBroadcast)
  for (trian,b) in c.contrib.dict
    val = a[trian]
    Base.materialize!(val,b)
  end
  a
end

Base.eltype(a::ArrayContribution) = eltype(first(values(a.dict)))
Base.eltype(a::Tuple{Vararg{ArrayContribution}}) = eltype(first(a))

function LinearAlgebra.fillstored!(a::ArrayContribution,v)
  for c in values(a.dict)
    LinearAlgebra.fillstored!(c,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::Tuple{Vararg{ArrayContribution}},v)
  map(a) do a
    LinearAlgebra.fillstored!(a,v)
  end
end
