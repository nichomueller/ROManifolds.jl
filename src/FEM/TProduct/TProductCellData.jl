"""
    TProductCellPoint{DS<:DomainStyle,A,B} <: CellDatum

"""
struct TProductCellPoint{DS<:DomainStyle,A<:CellPoint{DS},B<:AbstractVector{<:CellPoint{DS}}} <: CellDatum
  point::A
  single_points::B
end

get_tp_data(f::TProductCellPoint) = f.single_points

function CellData.get_triangulation(f::TProductCellPoint)
  s1 = first(f.single_points)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.single_points))
  trian
end

CellData.DomainStyle(::Type{<:TProductCellPoint{DS}}) where DS = DS()

# default behavior

function Arrays.evaluate!(cache,f::CellField,x::TProductCellPoint)
  evaluate!(cache,f,x.point)
end

"""
    TProductCellField{DS<:DomainStyle,A} <: CellField

"""
struct TProductCellField{DS<:DomainStyle,A} <: CellField
  single_fields::A
  domain_style::DS

  function TProductCellField(single_fields::A) where {A<:Vector{<:CellField}}
    @assert length(single_fields) > 0
    if any( map(i->DomainStyle(i)==ReferenceDomain(),single_fields) )
      domain_style = ReferenceDomain()
    else
      domain_style = PhysicalDomain()
    end
    DS = typeof(domain_style)
    new{DS,A}(single_fields,domain_style)
  end
end

get_tp_data(f::TProductCellField) = f.single_fields

function CellData.get_triangulation(f::TProductCellField)
  s1 = first(f.single_fields)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.single_fields))
  trian
end

CellData.DomainStyle(::Type{TProductCellField{DS}}) where DS = DS()

function LinearAlgebra.dot(a::TProductCellField,b::TProductCellField)
  @check length(a.single_fields) == length(b.single_fields)
  return sum(map(dot,a.single_fields,b.single_fields))
end

# gradients

"""
    TProductGradientCellField{A,B} <: CellField

"""
struct TProductGradientCellField{A,B} <: CellField
  cell_data::A
  gradient_cell_data::B
end

get_tp_data(a::TProductGradientCellField) = a.cell_data
CellData.DomainStyle(a::TProductGradientCellField) = DomainStyle(get_tp_data(a))
CellData.get_triangulation(a::TProductGradientCellField) = get_triangulation(get_tp_data(a))
get_gradient_data(a::TProductGradientCellField) = a.gradient_cell_data

(g::TProductGradientCellField)(x) = evaluate(g,x)

function Fields.gradient(f::TProductCellField)
  g = TProductCellField(gradient.(f.single_fields))
  return TProductGradientCellField(f,g)
end

# stores the evaluation of a TProductGradientCellField on a quadrature
struct TProductGradientEval{A,B,C}
  f::A
  g::B
  op::C
end

function TProductGradientEval(f::AbstractVector,g::AbstractVector)
  op = nothing
  TProductGradientEval(f,g,op)
end

get_tp_data(a::TProductGradientEval) = a.f
get_tp_gradient_data(a::TProductGradientEval) = a.g

# fe basis

struct TProductFEBasis{DS,BS,A,B} <: FEBasis
  basis::A
  trian::B
  domain_style::DS
  basis_style::BS
end

function TProductFEBasis(basis::Vector{<:FESpaces.FEBasis},trian::TProductTriangulation)
  b1 = testitem(basis)
  DS = DomainStyle(b1)
  BS = BasisStyle(b1)
  @check all(map(i -> DS===DomainStyle(i) && BS===BasisStyle(i),basis))
  TProductFEBasis(basis,trian,DS,BS)
end

function TProductFEBasis(basis::Vector{<:MultiField.MultiFieldCellField},trian::TProductTriangulation)
  b1 = testitem(basis)
  DS = DomainStyle(b1)
  BS = BasisStyle(first(b1))
  @check all(map(i -> DS===DomainStyle(i) && BS===BasisStyle(first(i)),basis))
  TProductFEBasis(basis,trian,DS,BS)
end

get_tp_data(f::TProductFEBasis) = f.basis
CellData.get_triangulation(f::TProductFEBasis) = f.trian
FESpaces.BasisStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = BS()
CellData.DomainStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = DS()

function Fields.gradient(f::TProductFEBasis)
  dbasis = map(gradient,f.basis)
  trian = get_triangulation(f)
  g = TProductFEBasis(dbasis,trian)
  return TProductGradientCellField(f,g)
end

const MultiFieldTProductFEBasis{DS,BS,B} = TProductFEBasis{DS,BS,Vector{MultiFieldCellField{DS}},B}

MultiField.num_fields(a::MultiFieldTProductFEBasis) = length(a.basis[1])

Base.length(a::MultiFieldTProductFEBasis) = num_fields(a)

function Base.getindex(a::MultiFieldTProductFEBasis,i::Integer)
  TProductFEBasis(getindex.(a.basis,i),a.trian,a.domain_style,a.basis_style)
end

function Base.iterate(a::MultiFieldTProductFEBasis,state=1)
  if state > num_fields(a)
    return nothing
  end
  astate = TProductFEBasis(getindex.(a.basis,state),a.trian,a.domain_style,a.basis_style)
  return astate,state+1
end

# evaluations

const TProductCellDatum = Union{TProductFEBasis,TProductCellField}

function Arrays.return_cache(f::TProductCellDatum,x::TProductCellPoint)
  @assert length(get_tp_data(f)) == length(get_tp_data(x))
  fitem = testitem(get_tp_data(f))
  xitem = testitem(get_tp_data(x))
  c1 = return_cache(fitem,xitem)
  fx1 = evaluate(fitem,xitem)
  cache = Vector{typeof(c1)}(undef,length(get_tp_data(f)))
  array = Vector{typeof(fx1)}(undef,length(get_tp_data(f)))
  return cache,array
end

function Arrays.evaluate!(_cache,f::TProductCellDatum,x::TProductCellPoint)
  cache,b = _cache
  @inbounds for i = eachindex(get_tp_data(f))
    b[i] = evaluate!(cache[i],get_tp_data(f)[i],get_tp_data(x)[i])
  end
  return b
end

function Arrays.return_cache(k::Operation,f::TProductCellDatum...)
  D = length(get_tp_data(first(f)))
  @assert all(map(fi -> length(get_tp_data(fi)) == D,f))
  fitem = map(testitem,get_tp_data.(f))
  c1 = return_cache(k,fitem...)
  Fill(c1,D)
end

function Arrays.evaluate!(cache,k::Operation,α::TProductCellDatum,β::TProductCellDatum)
  αβ = map(evaluate!,cache,Fill(k,length(cache)),get_tp_data(α),get_tp_data(β))
  TProductCellField(αβ)
end

function Arrays.return_cache(f::TProductGradientCellField,x::TProductCellPoint)
  cache = return_cache(get_tp_data(f),x)
  gradient_cache = return_cache(get_gradient_data(f),x)
  return cache,gradient_cache
end

function Arrays.evaluate!(_cache,f::TProductGradientCellField,x::TProductCellPoint)
  cache,gradient_cache = _cache
  fx = evaluate!(cache,get_tp_data(f),x)
  dfx = evaluate!(gradient_cache,get_gradient_data(f),x)
  return TProductGradientEval(fx,dfx)
end

function Arrays.return_cache(k::Operation,f::TProductGradientCellField...)
  cache = return_cache(k,map(get_tp_data,f)...)
  gradient_cache = return_cache(k,map(get_gradient_data,f)...)
  return cache,gradient_cache
end

function Arrays.evaluate!(_cache,k::Operation,α::TProductGradientCellField,β::TProductGradientCellField)
  cache,gradient_cache = _cache
  αβ = evaluate!(cache,k,get_tp_data(α),get_tp_data(β))
  dαβ = evaluate!(gradient_cache,k,get_gradient_data(α),get_gradient_data(β))
  return TProductGradientCellField(αβ,dαβ)
end

# integration

function CellData.integrate(f::TProductCellDatum,a::TProductMeasure)
  map(integrate,get_tp_data(f),a.measures_1d)
end

function CellData.integrate(f::TProductGradientCellField,a::TProductMeasure)
  fi = integrate(get_tp_data(f),a)
  dfi = integrate(get_gradient_data(f),a)
  TProductGradientEval(fi,dfi)
end

# this deals with a cell field +- a gradient cell field; for now, the result of
# these operations is simply the gradient cell field itself, since I'm only dealing with
# mass and stiffness matrices. See TProductArray and TProductGradientArray for
# more details. If in the future I consider more complicated structures (e.g.
# divergence/curl terms) this will have to change

for op in (:+,:-)
  @eval ($op)(a::AbstractArray,b::TProductGradientEval) = _add_tp_cell_data($op,a,b)
  @eval ($op)(a::TProductGradientEval,b::AbstractArray) = _add_tp_cell_data($op,a,b)
  @eval ($op)(a::TProductGradientEval,b::TProductGradientEval) = @notimplemented
end

Arrays.testitem(a::DomainContribution) = a[first([get_domains(a)...])]

function _add_tp_cell_data(op,a::AbstractVector{<:DomainContribution},b::TProductGradientEval)
  a1 = testitem(a[1])
  b1 = testitem(b.f[1])
  if isa(a1,AbstractArray{<:ArrayBlock})
    @check isa(b1,AbstractArray{<:ArrayBlock})
    if all(_is_different_block.(a,b.f))
      TProductGradientEval(op(a,b.f),b.g,op)
    else
      TProductGradientEval(b.f,b.g,op)
    end
  else
    TProductGradientEval(b.f,b.g,op)
  end
end

function _add_tp_cell_data(op,a::TProductGradientEval,b::AbstractVector{<:DomainContribution})
  a1 = testitem(a.f[1])
  b1 = testitem(b[1])
  if isa(b1,AbstractArray{<:ArrayBlock})
    @check isa(a1,AbstractArray{<:ArrayBlock})
    if all(_is_different_block.(a.f,b))
      TProductGradientEval(op(a.f,b),a.g,op)
    else
      TProductGradientEval(a.f,a.g,op)
    end
  else
    TProductGradientEval(a.f,a.g,op)
  end
end

function _is_different_block(a::DomainContribution,b::DomainContribution)
  b1 = testitem(b)
  for ai in values(a.dict)
    if b1[1].touched == ai[1].touched
      return false
    end
  end
  return true
end
