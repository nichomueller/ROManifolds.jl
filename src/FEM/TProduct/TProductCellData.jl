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
    abstract type TProductCellField <: CellField end

Subtypes:
- [`GenericTProductCellField`](@ref)
- [`TProductDiffCellField`](@ref)

"""
abstract type TProductCellField <: CellField end

get_tp_data(f::AbstractVector{<:AbstractArray}) = f
get_tp_data(f::AbstractVector{<:CellField}) = f
get_tp_data(f::TProductCellField) = @abstractmethod

get_diff_data(f::TProductCellField) = @abstractmethod

get_tp_diff_data(f::AbstractVector{<:AbstractArray}) = f
get_tp_diff_data(f::AbstractVector{<:CellField}) = f
get_tp_diff_data(f::TProductCellField) = @abstractmethod

"""
    GenericTProductCellField{DS<:DomainStyle,A,B} <: TProductCellField

"""
struct GenericTProductCellField{DS<:DomainStyle,A,B} <: TProductCellField
  single_fields::A
  trian::B
  domain_style::DS

  function GenericTProductCellField(single_fields::A,trian::B) where {A<:Vector{<:CellField},B<:TProductTriangulation}
    @assert length(single_fields) > 0
    if any( map(i->DomainStyle(i)==ReferenceDomain(),single_fields) )
      domain_style = ReferenceDomain()
    else
      domain_style = PhysicalDomain()
    end
    DS = typeof(domain_style)
    new{DS,A,B}(single_fields,trian,domain_style)
  end
end

get_tp_data(f::GenericTProductCellField) = f.single_fields

CellData.get_triangulation(f::GenericTProductCellField) = f.trian

CellData.DomainStyle(::Type{<:GenericTProductCellField{DS}}) where DS = DS()

# function LinearAlgebra.dot(f::GenericTProductCellField,b::GenericTProductCellField)
#   @check length(f.single_fields) == length(b.single_fields)
#   return sum(map(dot,f.single_fields,b.single_fields))
# end

# differentiation

"""
    abstract type TProductDiffCellField <: TProductCellField end

"""
abstract type TProductDiffCellField <: TProductCellField end

"""
    GenericTProductDiffCellField{A,B} <: TProductDiffCellField

"""
struct GenericTProductDiffCellField{O,A,B} <: TProductDiffCellField
  op::O
  cell_data::A
  diff_cell_data::B
end

const GradientTProductCellField{A,B} = GenericTProductDiffCellField{typeof(gradient),A,B}
const DivergenceTProductCellField{A,B} = GenericTProductDiffCellField{typeof(divergence),A,B}

CellData.get_data(f::GenericTProductDiffCellField) = f.cell_data
get_diff_data(f::GenericTProductDiffCellField) = f.diff_cell_data

get_tp_data(f::GenericTProductDiffCellField) = get_tp_data(f.cell_data)
get_tp_diff_data(f::GenericTProductDiffCellField) = get_tp_data(f.diff_cell_data)

CellData.get_triangulation(f::GenericTProductDiffCellField) = get_triangulation(f.cell_data)

CellData.DomainStyle(f::GenericTProductDiffCellField) = DomainStyle(f.cell_data)

function Fields.gradient(f::TProductCellField)
  g = GenericTProductCellField(gradient.(f.single_fields),f.trian)
  return GenericTProductDiffCellField(gradient,f,g)
end

function Fields.divergence(f::TProductCellField)
  g = GenericTProductCellField(divergence.(f.single_fields),f.trian)
  return GenericTProductDiffCellField(divergence,f,g)
end

# stores the evaluation of a GenericTProductDiffCellField on a quadrature
struct GenericTProductDiffEval{O,A,B}
  op::O
  f::A
  g::B
end

const GradientTProductEval{A,B} = GenericTProductDiffEval{typeof(gradient),A,B}
const DivergenceTProductEval{A,B} = GenericTProductDiffEval{typeof(divergence),A,B}

CellData.get_data(f::GenericTProductDiffEval) = f.f
get_diff_data(f::GenericTProductDiffEval) = f.g

get_tp_data(f::GenericTProductDiffEval) = get_tp_data(f.f)
get_tp_diff_data(f::GenericTProductDiffEval) = get_tp_data(f.g)

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
  return GenericTProductDiffCellField(gradient,f,g)
end

function Fields.divergence(f::TProductFEBasis)
  dbasis = map(divergence,f.basis)
  g = GenericTProductCellField(dbasis,f.trian)
  return GenericTProductDiffCellField(divergence,f,g)
end

const MultiFieldTProductFEBasis{DS,BS,B} = TProductFEBasis{DS,BS,Vector{MultiFieldCellField{DS}},B}

MultiField.num_fields(f::MultiFieldTProductFEBasis) = length(f.basis[1])

Base.length(f::MultiFieldTProductFEBasis) = num_fields(f)

function Base.getindex(f::MultiFieldTProductFEBasis,i::Integer)
  TProductFEBasis(getindex.(f.basis,i),f.trian,f.domain_style,f.basis_style)
end

function Base.iterate(f::MultiFieldTProductFEBasis,state=1)
  if state > num_fields(f)
    return nothing
  end
  astate = TProductFEBasis(getindex.(f.basis,state),f.trian,f.domain_style,f.basis_style)
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
  GenericTProductCellField(αβ,get_triangulation(α))
end

function Arrays.return_cache(f::GenericTProductDiffCellField,x::TProductCellPoint)
  cache = return_cache(get_data(f),x)
  diff_cache = return_cache(get_diff_data(f),x)
  return cache,diff_cache
end

function Arrays.evaluate!(_cache,f::GenericTProductDiffCellField,x::TProductCellPoint)
  cache,diff_cache = _cache
  fx = evaluate!(cache,get_data(f),x)
  dfx = evaluate!(diff_cache,get_diff_data(f),x)
  return GenericTProductDiffEval(f.op,fx,dfx)
end

function Arrays.return_cache(k::Operation,α::GradientTProductCellField,β::GradientTProductCellField)
  cache = return_cache(k,get_data(α),get_data(β))
  diff_cache = return_cache(k,get_diff_data(α),get_diff_data(β))
  return cache,diff_cache
end

function Arrays.return_cache(k::Operation{typeof(*)},α::TProductCellDatum,β::DivergenceTProductCellField)
  cache = return_cache(k,α,get_data(β))
  diff_cache = return_cache(k,α,get_diff_data(β))
  return cache,diff_cache
end

function Arrays.return_cache(k::Operation{typeof(*)},α::DivergenceTProductCellField,β::TProductCellDatum)
  cache = return_cache(k,get_data(α),β)
  diff_cache = return_cache(k,α,get_diff_data(β),β)
  return cache,diff_cache
end

function Arrays.evaluate!(_cache,k::Operation,α::GradientTProductCellField,β::GradientTProductCellField)
  cache,diff_cache = _cache
  αβ = evaluate!(cache,k,get_data(α),get_data(β))
  dαβ = evaluate!(diff_cache,k,get_diff_data(α),get_diff_data(β))
  return GenericTProductDiffCellField(gradient,αβ,dαβ)
end

function Arrays.evaluate!(_cache,k::Operation{typeof(*)},α::TProductCellDatum,β::DivergenceTProductCellField)
  cache,diff_cache = _cache
  αβ = evaluate!(cache,k,α,get_data(β))
  dαβ = evaluate!(cache,k,α,get_diff_data(β))
  return GenericTProductDiffCellField(divergence,αβ,dαβ)
end

function Arrays.evaluate!(_cache,k::Operation{typeof(*)},α::DivergenceTProductCellField,β::TProductCellDatum)
  cache,diff_cache = _cache
  αβ = evaluate!(cache,k,get_data(α),β)
  dαβ = evaluate!(cache,k,get_diff_data(α),β)
  return GenericTProductDiffCellField(divergence,αβ,dαβ)
end

# integration

function CellData.integrate(f::TProductCellDatum,a::TProductMeasure)
  map(integrate,get_tp_data(f),a.measures_1d)
end

function CellData.integrate(f::GenericTProductDiffCellField,a::TProductMeasure)
  fi = map(integrate,get_tp_data(f),a.measures_1d)
  dfi = map(integrate,get_tp_diff_data(f),a.measures_1d)
  GenericTProductDiffEval(f.op,fi,dfi)
end

for op in (:+,:-)
  @eval ($op)(a::AbstractArray,b::GenericTProductDiffEval) = GenericTProductDiffEval(b.op,$op(a,b.f),b.g)
  @eval ($op)(a::GenericTProductDiffEval,b::AbstractArray) = GenericTProductDiffEval(a.op,$op(a.f,b),a.g)
  @eval ($op)(a::GenericTProductDiffEval,b::GenericTProductDiffEval) = @notimplemented
end
