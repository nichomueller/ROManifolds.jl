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

function CellData.add_contribution!(
  a::PTDomainContribution,
  trian::Triangulation,
  b::PTArray,
  op=+)

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
  ptb = AffinePTArray(b,n)
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

for T in (:NonaffinePTArray,:AffinePTArray)
  @eval begin
    function CellData.move_contributions(scell_to_val::$T,args...)
      ptcell_mat_trian = map(scell_to_val.array) do x
        move_contributions(x,args...)
      end
      cell_to_val = $T(first.(ptcell_mat_trian))
      trian = first(last.(ptcell_mat_trian))
      cell_to_val,trian
    end
  end
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
  cont
end

function Arrays.testitem(a::DomainContribution)
  a
end

function Arrays.testitem(a::PTDomainContribution)
  b = DomainContribution()
  for (trian,array) in a.dict
    add_contribution!(b,trian,testitem(array))
  end
  b
end

# Interface to easily pass from one measure to another

struct PTIntegrand{T<:CellField}
  object::T
  meas::Measure
end

const ∫ₚ = PTIntegrand

function Arrays.getindex!(cont,a::PTIntegrand,meas::Measure)
  trian = get_triangulation(meas)
  itrian = get_triangulation(a.meas)
  if itrian == trian || is_parent(itrian,trian)
    integral = integrate(a.object,meas.quad)
    add_contribution!(cont,trian,integral)
    return cont
  end
  @unreachable """\n
    There is no contribution associated with the given mesh in this PTIntegrand object.
  """
end

function CellData.integrate(a::PTIntegrand)
  integrate(a.object,a.meas)
end

function CellData.integrate(a::PTIntegrand,meas::GenericMeasure...)
  @assert length(meas) == 1
  cont = init_contribution(a)
  for m in meas
    getindex!(cont,a,m)
  end
  cont
end

struct CollectionPTIntegrand{T,N}
  operations::NTuple{N,Union{typeof(+),typeof(-)}}
  integrands::NTuple{N,PTIntegrand{T}}
end

function Base.getindex(a::CollectionPTIntegrand,i::Int)
  a.operations[i],a.integrands[i]
end

function init_contribution(a...)
  PTDomainContribution()
end

function init_contribution(
  a::Union{PTIntegrand{<:OperationCellField},CollectionPTIntegrand{<:OperationCellField}}...)
  DomainContribution()
end

function Arrays.getindex!(cont,a::CollectionPTIntegrand{T,N} where T,meas::Measure) where N
  trian = get_triangulation(meas)
  for i = 1:N
    op,int = a[i]
    imeas = int.meas
    itrian = get_triangulation(imeas)
    if itrian == trian || is_parent(itrian,trian)
      integral = integrate(int.object,meas.quad)
      add_contribution!(cont,trian,integral,op)
    end
  end
  if num_domains(cont) > 0
    return cont
  end
  @unreachable """\n
    There is no contribution associated with the given mesh in this PTIntegrand object.
  """
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::PTIntegrand,b::PTIntegrand)
      CollectionPTIntegrand((+,$op),(a,b))
    end

    function ($op)(a::CollectionPTIntegrand,b::PTIntegrand)
      CollectionPTIntegrand((a.operations...,$op),(a.integrands...,b))
    end

    function ($op)(a::PTIntegrand,b::CollectionPTIntegrand)
      CollectionPTIntegrand(($op,b.operations...),(a,b.integrands...))
    end

    function ($op)(a::CollectionPTIntegrand,b::CollectionPTIntegrand)
      operations = (a.operations...,b.operations...)
      integrands = (a.integrands...,b.integrands...)
      CollectionPTIntegrand(operations,integrands)
    end
  end
end

function CellData.integrate(a::CollectionPTIntegrand{T,N} where T) where N
  cont = init_contribution(a)
  for i = 1:N
    op,int = a[i]
    imeas = int.meas
    itrian = get_triangulation(imeas)
    integral = integrate(int.object,imeas.quad)
    add_contribution!(cont,itrian,integral,op)
  end
  cont
end

function CellData.integrate(a::CollectionPTIntegrand,meas::GenericMeasure...)
  cont = init_contribution(a)
  for m in meas
    getindex!(cont,a,m)
  end
  cont
end

# Interface that allows to entirely eliminate terms from the (PT)DomainContribution

for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(::Nothing,::Nothing) = nothing
    ($op)(::Any,::Nothing) = nothing
    ($op)(::Nothing,::Any) = nothing
  end
end

Base.adjoint(::Nothing) = nothing
Base.broadcasted(f,a::Nothing,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Nothing,b::CellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::CellField,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
Fields.gradient(::Nothing) = nothing
LinearAlgebra.dot(::typeof(∇),::Nothing) = nothing
∂ₚt(::Nothing) = nothing

CellData.integrate(::Nothing,args...) = nothing

CellData.integrate(::Any,::Nothing) = nothing

for T in (:DomainContribution,:PTDomainContribution,:PTIntegrand,:CollectionPTIntegrand)
  @eval begin
    (+)(::Nothing,b::$T) = b
    (+)(a::$T,::Nothing) = a
    (-)(a::$T,::Nothing) = a
  end
end

function (-)(::Nothing,b::DomainContribution)
  for (trian,array) in b.dict
    b.dict[trian] = -array
  end
  b
end

function (-)(::Nothing,b::PTDomainContribution)
  for (trian,array) in b.dict
    b.dict[trian] = -array
  end
  b
end

function (-)(::Nothing,b::PTIntegrand)
  PTIntegrand(-b.object,b.meas)
end

function (-)(::Nothing,b::CollectionPTIntegrand)
  _neg_sign(::typeof(+)) = -
  _neg_sign(::typeof(-)) = +
  CollectionPTIntegrand(map(_neg_sign,b.operations),b.integrands)
end

function FESpaces.collect_cell_vector(::FESpace,::Nothing,args...)
  nothing
end

function FESpaces.collect_cell_matrix(::FESpace,::FESpace,::Nothing,args...)
  nothing
end
