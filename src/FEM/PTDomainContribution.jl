struct PTDomainContribution <: GridapType
  dict::IdDict{Triangulation,PTArray}
end

function PTDomainContribution()
  PTDomainContribution(IdDict{Triangulation,PTArray}())
end

CellData.get_triangulation(meas::Measure) = meas.quad.trian

CellData.num_domains(a::PTDomainContribution) = length(a.dict)

CellData.get_domains(a::PTDomainContribution) = keys(a.dict)

function CellData.get_contribution(a::PTDomainContribution,trian::Triangulation)
  if haskey(a.dict,trian) || haskey(a.dict,get_parent(trian))
     return a.dict[trian]
  elseif haskey(a.dict,get_parent(trian))
    return a.dict[get_parent(trian)]
  else
    @unreachable """\n
    There is no contribution associated with the given mesh in this PTDomainContribution object.
    """
  end
end

Base.getindex(a::PTDomainContribution,trian::Triangulation) = get_contribution(a,trian)

Base.sum(a::PTDomainContribution) = sum(map(sum,values(a.dict)))

Base.copy(a::PTDomainContribution) = PTDomainContribution(copy(a.dict))

function error_message(a::PTDomainContribution,b::Union{PTArray,AbstractArray})
  S = eltype(b)
  message = """\n
  You are trying to add a contribution with eltype $(S).
  Only cell-wise matrices, vectors, or numbers are accepted.

  Make sure that you are defining the terms in your weak form correctly.
  """
  if !(S<:AbstractMatrix || S<:AbstractVector || S<:Number || S<:ArrayBlock)
    @unreachable message
  end

  if length(a.dict) > 0
    T = eltype(first(values(a.dict)))
    if T <: AbstractMatrix || S<:(ArrayBlock{A,2} where A)
      @assert S<:AbstractMatrix || S<:(ArrayBlock{A,2} where A) message
    elseif T <: AbstractVector || S<:(ArrayBlock{A,1} where A)
      @assert S<:AbstractVector || S<:(ArrayBlock{A,1} where A) message
    elseif T <: Number
      @assert S<:Number message
    end
  end
end

function CellData.add_contribution!(
  a::PTDomainContribution,
  trian::Triangulation,
  b::PTArray,
  op=+)

  error_message(a,b)

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

function CellData.add_contribution!(
  a::PTDomainContribution,
  trian::Triangulation,
  b::AbstractArray,
  op=+)

  n = 0
  for atrian in get_domains(a)
    n = length(a[atrian])
    break
  end
  ptb = PTArray(b,n)
  add_contribution!(a,trian,ptb,op)
end

function (+)(a::PTDomainContribution,b::Union{PTDomainContribution,DomainContribution})
  c = copy(a)
  for (trian,array) in b.dict
    add_contribution!(c,trian,array)
  end
  c
end

function (-)(a::PTDomainContribution,b::Union{PTDomainContribution,DomainContribution})
  c = copy(a)
  for (trian,array) in b.dict
    add_contribution!(c,trian,array,-)
  end
  c
end

function (+)(a::DomainContribution,b::PTDomainContribution)
  (+)(b,a)
end

function (-)(a::DomainContribution,b::PTDomainContribution)
  c = (-)(b,a)
  for (trian,_) in c.dict
    c.dict[trian] = lazy_map(Broadcasting(-),c.dict[trian])
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

function CellData.move_contributions(scell_to_val::PTArray,args...)
  ptcell_mat_trian = map(scell_to_val.array) do x
    move_contributions(x,args...)
  end
  cell_to_val = PTArray(first.(ptcell_mat_trian))
  trian = first(last.(ptcell_mat_trian))
  cell_to_val,trian
end

function CellData.integrate(f::PTCellField,b::CellData.GenericMeasure)
  c = integrate(f,b.quad)
  cont = PTDomainContribution()
  add_contribution!(cont,b.quad.trian,c)
  cont
end

function CellData.integrate(f::PTCellField,b::CellData.CompositeMeasure)
  ic = integrate(f,b.quad)
  cont = PTDomainContribution()
  tc = move_contributions(ic,b.itrian,b.ttrian)
  add_contribution!(cont,b.quad.trian,tc)
  return cont
end
