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
  if haskey(a.dict,trian)
    return a.dict[trian]
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

struct PTIntegrand{T<:CellField}
  object::T
  meas::Measure
  PTIntegrand(object::T,meas::Measure) where T = new{T}(object,meas)
end

const ∫ₚ = PTIntegrand

function CellData.get_contribution(int::PTIntegrand,meas::Measure)
  trian = get_triangulation(meas)
  itrian = get_triangulation(int.meas)
  if itrian == trian || is_parent(itrian,trian)
    return integrate(int.object,meas)
  end
  @unreachable """\n
    There is no contribution associated with the given mesh in this PTIntegrand object.
  """
end

Base.getindex(int::PTIntegrand,meas::Measure) = get_contribution(int,meas)

function CellData.integrate(int::PTIntegrand,args...)
  cont = DomainContribution()
  integral = integrate(int.object,int.meas)
  trian = get_triangulation(int.meas)
  add_contribution!(cont,trian,integral[trian])
end

function CellData.integrate(int::PTIntegrand{<:PTCellField},args...)
  cont = PTDomainContribution()
  integral = integrate(int.object,int.meas)
  trian = get_triangulation(int.meas)
  add_contribution!(cont,trian,integral[trian])
end

struct CollectionPTIntegrand
  dict::IdDict{Function,Tuple{Vararg{PTIntegrand}}}
end

function CollectionPTIntegrand()
  CollectionPTIntegrand(IdDict{Function,Tuple{Vararg{PTIntegrand}}}())
end

function CellData.get_contribution(a::CollectionPTIntegrand,meas::Measure,cont=PTDomainContribution())
  trian = get_triangulation(meas)
  for (op,int) in a.dict
    for i in int
      itrian = get_triangulation(i.meas)
      if itrian == trian || is_parent(itrian,trian)
        integral = integrate(i.object,meas)
        add_contribution!(cont,trian,integral[trian],op)
      end
    end
  end
  if num_domains(cont) > 0
    return cont
  end
  @unreachable """\n
    There is no contribution associated with the given mesh in this PTIntegrand object.
  """
end

Base.getindex(int::CollectionPTIntegrand,meas::Measure) = get_contribution(int,meas)

function CellData.add_contribution!(a::CollectionPTIntegrand,op::Function,b::PTIntegrand)
  if haskey(a.dict,op)
    a.dict[op] = (a.dict[op]...,b)
  else
    a.dict[op] = (b,)
  end
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::PTIntegrand,b::PTIntegrand)
      c = CollectionPTIntegrand()
      add_contribution!(c,+,a)
      add_contribution!(c,$op,b)
      c
    end

    function ($op)(a::CollectionPTIntegrand,b::PTIntegrand)
      add_contribution!(a,$op,b)
      a
    end

    function ($op)(a::PTIntegrand,b::CollectionPTIntegrand)
      c = CollectionPTIntegrand()
      add_contribution!(c,+,a)
      for (_op,int) in b.dict
        if (&)($op == +,_op == +) || (&)($op == -,_op == -)
          add_contribution!(c,+,int)
        else
          add_contribution!(c,-,int)
        end
      end
      c
    end

    function ($op)(a::CollectionPTIntegrand,b::CollectionPTIntegrand)
      for (_op,int) in b.dict
        if (&)($op == +,_op == +) || (&)($op == -,_op == -)
          add_contribution!(a,+,int)
        else
          add_contribution!(a,-,int)
        end
      end
      a
    end
  end
end

function CellData.integrate(a::CollectionPTIntegrand,cont=PTDomainContribution())
  for (op,int) in a.dict
    for i in int
      itrian = get_triangulation(i.meas)
      integral = integrate(i)
      add_contribution!(cont,itrian,integral[itrian],op)
    end
  end
  cont
end

for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(::Nothing,::Nothing) = nothing
    ($op)(::Any,::Nothing) = nothing
    ($op)(::Nothing,::Any) = nothing
  end
end

Base.broadcasted(f,a::Nothing,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Nothing,b::CellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::CellField,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)

Fields.gradient(::Nothing) = nothing
LinearAlgebra.dot(::typeof(∇),::Nothing) = nothing

CellData.integrate(::Nothing,::GenericMeasure) = nothing

CellData.integrate(::Nothing,::CellQuadrature) = nothing

CellData.integrate(::Any,::Nothing) = nothing

for T in (:DomainContribution,:PTDomainContribution)
  @eval begin
    (+)(::Nothing,b::$T) = b

    (+)(a::$T,::Nothing) = a

    function (-)(::Nothing,b::$T)
      for (trian,array) in b.dict
        b.dict[trian] = -array
      end
      b
    end

    (-)(a::$T,::Nothing) = a
  end
end

function FESpaces.collect_cell_vector(test,::Nothing,args...)
  nothing
end

function FESpaces.collect_cell_matrix(trial,test,::Nothing,args...)
  nothing
end
