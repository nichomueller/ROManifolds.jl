struct RBAlgebraicContribution
  dict::IdDict{Triangulation,AbstractArray}
end

RBAlgebraicContribution() = RBAlgebraicContribution(IdDict{Triangulation,AbstractArray}())

Gridap.CellData.num_domains(a::RBAlgebraicContribution) = length(a.dict)

Gridap.CellData.get_domains(a::RBAlgebraicContribution) = keys(a.dict)

function Gridap.CellData.get_contribution(
  a::RBAlgebraicContribution,
  trian::Triangulation)

  if haskey(a.dict,trian)
    return a.dict[trian]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this RBAlgebraicContribution object.
    """
  end
end

Base.getindex(a::RBAlgebraicContribution,trian::Triangulation) = get_contribution(a,trian)

function Gridap.CellData.add_contribution!(
  a::RBAlgebraicContribution,
  trian::Triangulation,
  b::AbstractArray)

  @check !haskey(a.dict,trian)
  a.dict[trian] = b
  a
end

function get_measures(a::RBAlgebraicContribution,degree::Int)
  numd = num_domains(a)
  meas = Vector{Measure}(undef,numd)
  for (itrian,trian) in enumerate(get_domains(a))
    meas[itrian] = Measure(trian,degree)
  end
  meas
end
