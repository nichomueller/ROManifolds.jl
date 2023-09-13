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
  dict::IdDict{Triangulation,Union{PTArray,AbstractArray}}
end

function PTDomainContribution()
  PTDomainContribution(IdDict{Triangulation,Union{PTArray,AbstractArray}}())
end

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
  b,op=+)

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

Base.sum(a::PTDomainContribution) = sum(map(sum,values(a.dict)))

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
  c = ptintegrate(f,b.quad)
  cont = PTDomainContribution()
  add_contribution!(cont,b.quad.trian,c)
  cont
end

function ptintegrate(f,b::CellData.CompositeMeasure)
  ic = ptintegrate(f,b.quad)
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

function ptintegrate(f::CellField,quad::CellQuadrature)
  trian_f = get_triangulation(f)
  trian_x = get_triangulation(quad)

  msg = """\n
    Your are trying to integrate a CellField using a CellQuadrature defined on incompatible
    triangulations. Verify that either the two objects are defined in the same triangulation
    or that the triangulaiton of the CellField is the background triangulation of the CellQuadrature.
    """
  @check is_change_possible(trian_f,trian_x) msg

  b = change_domain(f,quad.trian,quad.data_domain_style)
  x = get_cell_points(quad)
  bx = b(x)
  if isaffine(bx)
    bx = testitem(bx)
  end
  if quad.data_domain_style == PhysicalDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    lazy_map(IntegrationMap(),bx,quad.cell_weight)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    cell_map = get_cell_map(quad.trian)
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == ReferenceDomain()
    cell_map = Fill(GenericField(identity),length(bx))
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
  else
    @notimplemented
  end
end
