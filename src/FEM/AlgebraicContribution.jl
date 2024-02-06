struct AlgebraicContribution <: GridapType
  dict::IdDict{Triangulation,AbstractArray}
end

AlgebraicContribution() = AlgebraicContribution(IdDict{Triangulation,AbstractArray}())

CellData.num_domains(a::AlgebraicContribution) = length(a.dict)

CellData.get_domains(a::AlgebraicContribution) = keys(a.dict)

function CellData.get_contribution(a::AlgebraicContribution,trian::Triangulation)
  if haskey(a.dict,trian)
     return a.dict[trian]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this AlgebraicContribution object.
    """
  end
end

Base.getindex(a::AlgebraicContribution,trian::Triangulation) = get_contribution(a,trian)

function CellData.add_contribution!(a::AlgebraicContribution,b::AbstractArray,trian::Triangulation)
  if haskey(a.dict,trian)
    @notimplemented
  else
    a.dict[trian] = b
  end
  a
end

Base.setindex!(a::AlgebraicContribution,b::AbstractArray,trian::Triangulation) = add_contribution!(a,b,trian)

Base.eltype(a::AlgebraicContribution) = eltype(first(values(a.dict)))
Base.eltype(a::Tuple{Vararg{AlgebraicContribution}}) = eltype(first(a))

function LinearAlgebra.fillstored!(a::AlgebraicContribution,v)
  for c in values(a.dict)
    LinearAlgebra.fillstored!(c,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::Tuple{Vararg{AlgebraicContribution}},v)
  map(a) do a
    LinearAlgebra.fillstored!(a,v)
  end
end
