struct PIntegrand
  object::CellField
  meas::Measure
end

const ∫ₚ = PIntegrand

function Arrays.evaluate(int::PIntegrand)
  obj = int.object
  meas = int.meas
  ptintegrate(obj,meas)
end

struct PTDomainContribution <: GridapType
  dict::IdDict{Triangulation,PTArray}
end

PTDomainContribution() = PTDomainContribution(IdDict{Triangulation,PTArray}())

CellData.num_domains(a::PTDomainContribution) = length(a.dict)

CellData.get_domains(a::PTDomainContribution) = keys(a.dict)

function CellData.get_contribution(a::PTDomainContribution,trian::Triangulation)
  if haskey(a.dict,trian)
     return a.dict[trian]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this PTDomainContribution object.
    """
  end
end

Base.getindex(a::PTDomainContribution,trian::Triangulation) = get_contribution(a,trian)

function CellData.add_contribution!(
  a::PTDomainContribution,
  trian::Triangulation,
  b::PTArray{P},
  op=+) where P

  S = eltype(P)
  if !(S<:AbstractMatrix || S<:AbstractVector || S<:Number || S<:ArrayBlock)
    @unreachable """\n
    You are trying to add a contribution with eltype $(S).
    Only cell-wise matrices, vectors, or numbers are accepted.

    Make sure that you are defining the terms in your weak form correctly.
    """
  end

  if length(a.dict) > 0
    T = eltype(first(values(a.dict)))
    if T <: AbstractMatrix || S<:(ArrayBlock{A,2} where A)
      @assert S<:AbstractMatrix || S<:(ArrayBlock{A,2} where A) """\n
      You are trying to add a contribution with eltype $(S) to a PTDomainContribution that
      stores cell-wise matrices.

      Make sure that you are defining the terms in your weak form correctly.
      """
    elseif T <: AbstractVector || S<:(ArrayBlock{A,1} where A)
      @assert S<:AbstractVector || S<:(ArrayBlock{A,1} where A) """\n
      You are trying to add a contribution with eltype $(S) to a PTDomainContribution that
      stores cell-wise vectors.

      Make sure that you are defining the terms in your weak form correctly.
      """
    elseif T <: Number
      @assert S<:Number """\n
      You are trying to add a contribution with eltype $(S) to a PTDomainContribution that
      stores cell-wise numbers.

      Make sure that you are defining the terms in your weak form correctly.
      """
    end
  end

  if haskey(a.dict,trian)
    a.dict[trian] = lazy_map(Broadcasting(op),a.dict[trian],b)
  else
    if op == +
      a.dict[trian] = b
    else
      a.dict[trian] = lazy_map(Broadcasting(op),b)
    end
  end
  a
end

function Base.sum(a::PTDomainContribution)
  ptarrays = values(a.dict).array
  map(x->sum(map(sum,x)),ptarrays)
end

Base.copy(a::PTDomainContribution) = PTDomainContribution(copy(a.dict))

function (+)(a::PTDomainContribution,b::PTDomainContribution)
  c = copy(a)
  for (trian,array) in b.dict
    add_contribution!(c,trian,array)
  end
  c
end

function (-)(a::PTDomainContribution,b::PTDomainContribution)
  c = copy(a)
  for (trian,array) in b.dict
    add_contribution!(c,trian,array,-)
  end
  c
end

function (*)(a::Number,b::PTDomainContribution)
  c = PTDomainContribution()
  for (trian,array_old) in b.dict
    s = size(get_cell_map(trian))
    array_new = lazy_map(Broadcasting(*),Fill(a,s),array_old)
    add_contribution!(c,trian,array_new)
  end
  c
end

(*)(a::PTDomainContribution,b::Number) = b*a

function CellData.get_array(a::PTDomainContribution)
  @assert num_domains(a) == 1 """\n
  Method get_array(a::PTDomainContribution) can be called only
  when the PTDomainContribution object involves just one domain.
  """
  a.dict[first(keys(a.dict))]
end

function ptintegrate(f,b::CellData.GenericMeasure)
  c = integrate(f,b.quad)
  cont = PTDomainContribution()
  add_contribution!(cont,b.quad.trian,c)
  cont
end

function ptintegrate(f,b::CellData.CompositeMeasure)
  ic = integrate(f,b.quad)
  cont = PTDomainContribution()
  tc = move_contributions(ic,b.itrian,b.ttrian)
  add_contribution!(cont,b.ttrian,tc)
  return cont
end

function CellData.move_contributions(scell_to_val::PTArray,args...)
  y = map(x->move_contributions(x,args...),scell_to_val.array)
  ptcell_mat = PTArray(first.(y))
  trian = first(last.(y))
  ptcell_mat,trian
end
