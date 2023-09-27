struct PTDomainContribution <: GridapType
  dict::IdDict{Measure,Union{PTArray,AbstractArray}}
end

function PTDomainContribution()
  PTDomainContribution(IdDict{Measure,Union{PTArray,AbstractArray}}())
end

CellData.get_triangulation(meas::Measure) = meas.quad.trian

CellData.num_domains(a::PTDomainContribution) = length(a.dict)

CellData.get_domains(a::PTDomainContribution) = keys(a.dict)

function CellData.get_contribution(a::PTDomainContribution,meas::Measure)
  if haskey(a.dict,meas) || haskey(a.dict,get_parent(meas))
     return a.dict[meas]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this PTDomainContribution object.
    """
  end
end

function CellData.get_contribution(a::PTDomainContribution,trian::Triangulation)
  for meas in get_domains(a)
    mtrian = get_triangulation(meas)
    if is_parent(trian,mtrian) || trian == mtrian
      get_contribution(a,meas)
    end
  end
  @unreachable """\n
    There is not contribution associated with the given mesh in this PTDomainContribution object.
    """
end

Base.getindex(a::PTDomainContribution,meas::Measure) = get_contribution(a,meas)

function CellData.add_contribution!(
  a::PTDomainContribution,
  meas::Measure,
  b::Union{PTArray,AbstractArray},
  op=+)

  S = eltype(b)
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

  if haskey(a.dict,meas)
    a.dict[meas] = lazy_map(Broadcasting(op),a.dict[meas],b)
  else
    if op == +
      a.dict[meas] = b
    else
      a.dict[meas] = lazy_map(Broadcasting(op),b)
    end
  end
  a
end

Base.sum(a::PTDomainContribution) = sum(map(sum,values(a.dict)))

Base.copy(a::PTDomainContribution) = PTDomainContribution(copy(a.dict))

function (+)(a::PTDomainContribution,b::PTDomainContribution)
  c = copy(a)
  for (meas,array) in b.dict
    add_contribution!(c,meas,array)
  end
  c
end

function (-)(a::PTDomainContribution,b::PTDomainContribution)
  c = copy(a)
  for (meas,array) in b.dict
    add_contribution!(c,meas,array,-)
  end
  c
end

function (*)(a::Number,b::PTDomainContribution)
  c = PTDomainContribution()
  for (meas,array_old) in b.dict
    trian = get_triangulation(meas)
    s = size(get_cell_map(trian))
    array_new = lazy_map(Broadcasting(*),Fill(a,s),array_old)
    add_contribution!(c,meas,array_new)
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

function CellData.move_contributions(scell_to_val::PTArray,args...)
  ptcell_mat_trian = map(scell_to_val.array) do x
    move_contributions(x,args...)
  end
  cell_to_val = PTArray(first.(ptcell_mat_trian))
  trian = first(last.(ptcell_mat_trian))
  cell_to_val,trian
end

function ptintegrate(f::CellField,b::CellData.GenericMeasure)
  c = integrate(f,b.quad)
  cont = PTDomainContribution()
  add_contribution!(cont,b,c)
  cont
end

function ptintegrate(f::CellField,b::CellData.CompositeMeasure)
  ic = integrate(f,b.quad)
  cont = PTDomainContribution()
  tc = move_contributions(ic,b.itrian,b.ttrian)
  add_contribution!(cont,b,tc)
  return cont
end

const ∫ₚ = ptintegrate
