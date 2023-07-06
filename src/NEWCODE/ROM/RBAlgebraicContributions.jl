struct RBAlgebraicContribution{N}
  dict::IdDict{Triangulation,Array{RBAffineDecompositions,N}}
end

function RBResidualContribution()
  RBAlgebraicContribution{1}(IdDict{Triangulation,Vector{RBAffineDecompositions}}())
end

function RBJacobianContribution()
  RBAlgebraicContribution{2}(IdDict{Triangulation,Matrix{RBAffineDecompositions}}())
end

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
  a::RBAlgebraicContribution{N},
  trian::Triangulation,
  b::Array{<:RBAffineDecompositions,N}) where N

  @check !haskey(a.dict,trian)
  a.dict[trian] = b
  a
end

function get_measures(a::RBAlgebraicContribution)
  measures = Measures[]
  for (_,rbad) in a.dict
    push!(measures,rbad.integration_domain.meas)
  end
  unique(measures)
end
