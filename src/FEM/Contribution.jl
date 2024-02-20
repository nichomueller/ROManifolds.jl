struct Contribution{K,V} <: AbstractDict{K,V}
  dict::IdDict{K,V}
end

Base.length(a::Contribution) = length(a.dict)
Base.iterate(a::Contribution,i...) = iterate(a.dict,i...)
Base.getindex(a::Contribution{K},key::K) where K = get_contribution(a,key)
Base.setindex!(a::Contribution,val::V,key::K) where {K,V} = add_contribution!(a,val,key)
CellData.num_domains(a::Contribution) = length(a.dict)
CellData.get_domains(a::Contribution) = collect(keys(a.dict))
get_values(a::Contribution) = collect(values(a.dict))

function CellData.get_contribution(
  a::Contribution{K},
  key::K) where K

  if haskey(a.dict,key)
     return a.dict[key]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this $(typeof(a)) object.
    """
  end
end

function CellData.add_contribution!(
  a::Contribution{K,V},
  val::V,
  key::K) where {K,V}

  if haskey(a.dict,key)
    a.dict[key] += val
  else
    a.dict[key] = val
  end
  a
end

const ArrayContribution = Contribution{Triangulation,AbstractArray}

array_contribution() = Contribution(IdDict{Triangulation,AbstractArray}())

struct ContributionBroadcast{D}
  contrib::D
end

function Base.broadcasted(f,a::ArrayContribution,b::Number)
  c = Contribution(IdDict{Triangulation,FEM.AbstractParamBroadcast}())
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
