abstract type AbstractRBAlgebraicContribution{T} end

CellData.num_domains(a::AbstractRBAlgebraicContribution) = length(a.dict)
CellData.get_domains(a::AbstractRBAlgebraicContribution) = keys(a.dict)

function CellData.get_contribution(
  a::AbstractRBAlgebraicContribution,
  meas::Measure)

  if haskey(a.dict,meas)
    return a.dict[meas]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this RBAlgebraicContribution object.
    """
  end
end

Base.getindex(a::AbstractRBAlgebraicContribution,meas::Measure) = get_contribution(a,meas)

function CellData.add_contribution!(
  a::AbstractRBAlgebraicContribution,
  meas::Measure,
  b)

  @check !haskey(a.dict,meas)
  a.dict[meas] = b
  a
end

struct RBAlgebraicContribution{T} <: AbstractRBAlgebraicContribution{T}
  dict::IdDict{Measure,RBAffineDecompositions{T}}
  function RBAlgebraicContribution(::Type{T}) where T
    new{T}(IdDict{Measure,RBAffineDecompositions{T}}())
  end
end

struct RBBlockAlgebraicContribution{T} <: AbstractRBAlgebraicContribution{T}
  dict::IdDict{Measure,RBBlockAffineDecompositions{T}}
  function RBBlockAlgebraicContribution(::Type{T}) where T
    new{T}(IdDict{Measure,RBBlockAlgebraicContributions{T}}())
  end
end
