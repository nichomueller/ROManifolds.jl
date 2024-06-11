struct TProductCellPoint{DS<:DomainStyle,A,B} <: CellDatum
  point::A
  single_points::B
  function TProductCellPoint(
    point::A,single_points::B
    ) where {DS,A<:CellPoint{DS},B<:AbstractVector{<:CellPoint{DS}}}
    new{DS,A,B}(point,single_points)
  end
end

CellData.get_data(f::TProductCellPoint) = f.point
Base.length(a::TProductCellPoint) = length(a.single_points)

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

CellData.get_data(f::TProductCellField) = f.single_fields

function CellData.get_triangulation(f::TProductCellField)
  s1 = first(f.single_fields)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.single_fields))
  trian
end

CellData.DomainStyle(::Type{TProductCellField{DS}}) where DS = DS()
Base.length(a::TProductCellField) = length(a.single_fields)

function LinearAlgebra.dot(a::TProductCellField,b::TProductCellField)
  @check length(a) == length(b)
  return sum(map(dot,a.single_fields,b.single_fields))
end

# gradients

struct TProductGradientCellField{A,B} <: CellField
  cell_data::A
  gradient_cell_data::B
end

CellData.get_data(a::TProductGradientCellField) = a.cell_data
CellData.DomainStyle(a::TProductGradientCellField) = DomainStyle(get_data(a))
CellData.get_triangulation(a::TProductGradientCellField) = get_triangulation(get_data(a))
get_gradient_data(a::TProductGradientCellField) = a.gradient_cell_data

(g::TProductGradientCellField)(x) = evaluate(g,x)

function Fields.gradient(f::TProductCellField)
  g = TProductCellField(gradient.(f.single_fields))
  return TProductGradientCellField(f,g)
end

function Fields.gradient(f::TProductFEBasis)
  dbasis = map(gradient,f.basis)
  trian = get_triangulation(f)
  g = TProductFEBasis(dbasis,trian)
  return TProductGradientCellField(f,g)
end

struct TProductGradientEval{A,B,C}
  f::A
  g::B
  op::C
end

function TProductGradientEval(f::AbstractVector,g::AbstractVector)
  op = nothing
  TProductGradientEval(f,g,op)
end

CellData.get_data(a::TProductGradientEval) = a.f
get_gradient_data(a::TProductGradientEval) = a.g

# evaluations

const TProductCellDatum = Union{TProductFEBasis,TProductCellField}

function Arrays.return_cache(f::TProductCellDatum,x::TProductCellPoint)
  @assert length(f) == length(x)
  fitem = testitem(get_data(f))
  xitem = testitem(get_data(x))
  c1 = return_cache(fitem,xitem)
  fx1 = evaluate(fitem,xitem)
  cache = Vector{typeof(c1)}(undef,length(f))
  array = Vector{typeof(fx1)}(undef,length(f))
  return cache,array
end

function Arrays.evaluate!(_cache,f::TProductCellDatum,x::TProductCellPoint)
  cache,b = _cache
  @inbounds for i = 1:length(f)
    b[i] = evaluate!(cache[i],get_data(f)[i],get_data(x)[i])
  end
  return b
end

function Arrays.return_cache(k::Operation,f::TProductCellDatum...)
  D = length(first(f))
  @assert all(map(i -> length(get_data(i)) == D,f))
  fitem = map(testitem,get_data.(f))
  c1 = return_cache(k,fitem...)
  Fill(c1,D),Fill(k,D)
end

function Arrays.evaluate!(_cache,k::Operation,α::TProductCellDatum,β::TProductCellDatum)
  cache,K = _cache
  αβ = map(evaluate!,cache,K,get_data(α),get_data(β))
  TProductCellField(αβ)
end

function Arrays.return_cache(f::TProductGradientCellField,x::TProductCellPoint)
  cache = return_cache(get_data(f),x)
  gradient_cache = return_cache(get_gradient_data(f),x)
  return cache,gradient_cache
end

function Arrays.evaluate!(_cache,f::TProductGradientCellField,x::TProductCellPoint)
  cache,gradient_cache = _cache
  fx = evaluate!(cache,get_data(f),x)
  dfx = evaluate!(gradient_cache,get_gradient_data(f),x)
  return TProductGradientEval(fx,dfx)
end

function Arrays.return_cache(k::Operation,f::TProductGradientCellField...)
  cache = return_cache(k,map(get_data,f)...)
  gradient_cache = return_cache(k,map(get_gradient_data,f)...)
  return cache,gradient_cache
end

function Arrays.evaluate!(_cache,k::Operation,α::TProductGradientCellField,β::TProductGradientCellField)
  cache,gradient_cache = _cache
  αβ = evaluate!(cache,k,get_data(α),get_data(β))
  dαβ = evaluate!(gradient_cache,k,get_gradient_data(α),get_gradient_data(β))
  return TProductGradientCellField(αβ,dαβ)
end

# integration

function CellData.integrate(f::TProductCellDatum,a::TProductMeasure)
  map(integrate,get_data(f),a.measures_1d)
end

function CellData.integrate(f::TProductGradientCellField,a::TProductMeasure)
  fi = integrate(get_data(f),a)
  dfi = integrate(get_gradient_data(f),a)
  TProductGradientEval(fi,dfi)
end

# fe basis

struct TProductFEBasis{DS,BS,A,B} <: FEBasis
  basis::A
  trian::B
  domain_style::DS
  basis_style::BS
end

function TProductFEBasis(basis::Vector,trian::TProductTriangulation)
  b1 = testitem(basis)
  DS = DomainStyle(b1)
  BS = BasisStyle(b1)
  @check all(map(i -> DS===DomainStyle(i) && BS===BasisStyle(i),basis))
  TProductFEBasis(basis,trian,DS,BS)
end

CellData.get_data(f::TProductFEBasis) = f.basis
CellData.get_triangulation(f::TProductFEBasis) = f.trian
FESpaces.BasisStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = BS
CellData.DomainStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = DS
