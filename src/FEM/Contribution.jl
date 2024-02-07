struct Contribution{T} <: AbstractDict{Triangulation,T}
  dict::IdDict{Triangulation,T}
end

Base.length(a::Contribution) = length(a.dict)
Base.iterate(a::Contribution,i...) = iterate(a.dict,i...)
Base.values(a::Contribution) = values(a.dict)
Base.keys(a::Contribution) = keys(a.dict)
CellData.num_domains(a::Contribution) = length(a)
CellData.get_domains(a::Contribution) = collect(keys(a))
get_values(a::Contribution) = collect(values(a))

function CellData.get_contribution(a::T,trian::Triangulation) where T<:Contribution
  if haskey(a.dict,trian)
     return a.dict[trian]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this $T object.
    """
  end
end

Base.getindex(a::Contribution,trian::Triangulation) = get_contribution(a,trian)

function CellData.add_contribution!(a::Contribution{T},b::T,trian::Triangulation) where T
  if haskey(a.dict,trian)
    @notimplemented
  else
    a.dict[trian] = b
  end
  a
end

Base.setindex!(a::Contribution,b,trian::Triangulation) = add_contribution!(a,b,trian)

const ArrayContribution = Contribution{AbstractArray}

array_contribution() = Contribution(IdDict{Triangulation,AbstractArray}())

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
