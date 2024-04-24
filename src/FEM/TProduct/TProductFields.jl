# struct KroneckerMap <: Map end

# Arrays.return_cache(k::KroneckerMap,cache::Tuple{Vararg{Any}}) = return_cache(k,first.(cache))
# Arrays.return_cache(k::KroneckerMap,cache::Tuple{Vararg{CachedArray}}) = return_cache(k,get_array.(cache))
# Arrays.return_cache(::KroneckerMap,cache::Tuple{Vararg{AbstractArray}}) = kron(cache...)

abstract type TPField <: Field end

struct GenericTPField{D,I,A} <: TPField
  factors::A
  isotropy::I
  function GenericTPField(factors::A,isotropy::I=Isotropy(factors)) where {I,A<:AbstractVector{<:Field}}
    D = length(factors)
    new{D,I,A}(factors,isotropy)
  end
end

function GenericTPField(factors::AbstractVector{<:AbstractVector{<:Field}},index::Int,args...)
  kindices = get_kindices(factors,index)
  ifactors = map(getindex,factors,kindices)
  GenericTPField(ifactors,args...)
end

get_factors(a::GenericTPField) = a.factors

function Arrays.return_cache(a::GenericTPField,x::AbstractTensorProductPoints)
  factors = get_factors(a)
  points = get_factors(x)
  cache = map(return_cache,factors,points)
  return cache
end

function Arrays.evaluate!(cache,a::GenericTPField,x::AbstractTensorProductPoints)
  factors = get_factors(a)
  points = get_factors(x)
  ax = map(evaluate!,cache,factors,points)
  return kronecker(ax...)
end

for op in (:(Fields.∇),:(Fields.∇∇))
  @eval begin
    function $op(a::GenericTPField)
      factors = Broadcasting($op)(get_factors(a))
      isotropy = get_isotropy(a)
      GenericTPField(factors,isotropy)
    end
  end
end

abstract type TensorProductField{D,I} <: AbstractVector{TPField} end

get_factors(::TensorProductField) = @abstractmethod
get_field(::TensorProductField) = @abstractmethod
get_isotropy(::TensorProductField{D,I} where D) where I = I()

for F in (:(TensorProductField),:(AbstractArray{<:TensorProductField}))
  @eval begin
    function Arrays.return_value(a::$F,x::Point)
      return_value(get_field(a),x)
    end
  end
  for T in (:(Point),:(AbstractArray{<:Point}))
    @eval begin
      function Arrays.return_cache(a::$F,x::$T)
        return_cache(get_field(a),x)
      end
      function Arrays.evaluate!(cache,a::$F,x::$T)
        evaluate!(cache,get_field(a),x)
      end
    end
  end
end

struct GenericTensorProductField{D,I,A} <: TensorProductField{D,I}
  factors::A
  isotropy::I
  function GenericTensorProductField(factors::A,isotropy::I=Isotropy(factors)) where {A,I}
    D = length(factors)
    new{D,I,A}(factors,field,isotropy)
  end
end

Base.length(a::GenericTensorProductField) = prod(length.(a.factors))
Base.size(a::GenericTensorProductField) = (length(a),)
Base.getindex(a::GenericTensorProductField,i::Integer...) = GenericTPField(a.field,i...)

get_factors(a::GenericTensorProductField) = a.factors
get_field(a::GenericTensorProductField) = a.field

_return_vec_cache(cache::Tuple{Vararg{Any}},D::Int) = _return_vec_cache(first(cache),D)
_return_vec_cache(cache::CachedArray,D::Int) = _return_vec_cache(get_array(cache),D)
_return_vec_cache(cache::AbstractArray,D::Int) = Vector{typeof(cache)}(undef,D)

function Arrays.testargs(f::Field,x::AbstractTensorProductPoints)
  (copy(x),)
end

function Arrays.return_cache(
  a::GenericTensorProductField{D},
  x::AbstractTensorProductPoints{D}
  ) where D

  factors = get_factors(a)
  points = get_factors(x)
  cache = return_cache(factors[1],points[1])
  r = _return_vec_cache(cache,D)
  return r,cache
end

function Arrays.evaluate!(
  _cache,
  a::GenericTensorProductField{D},
  x::AbstractTensorProductPoints{D}
  ) where D

  r,cache = _cache
  factors = get_factors(a)
  points = get_factors(x)
  indices_map = get_indices_map(x)
  @inbounds for d = 1:D
    r[d] = evaluate!(cache,factors[d],points[d])
  end
  tpr = FieldFactors(r,indices_map)
  return tpr
end

function Arrays.return_cache(
  a::GenericTensorProductField{Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  factors = get_factors(a)
  points = get_factors(x)
  return_cache(factors[1],points[1])
end

function Arrays.evaluate!(
  cache,
  a::GenericTensorProductField{Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  factors = get_factors(a)
  points = get_factors(x)
  indices_map = get_indices_map(x)
  r = evaluate!(cache,factors[1],points[1])
  tpr = FieldFactors(Fill(r,D),indices_map)
  return tpr
end

struct TensorProductAffineMap{D,I,T,L} <: TensorProductField{D,I}
  map::AffineMap{D,D,T,L}
  function TensorProductAffineMap(map::AffineMap{D,D,T,L}) where {D,T,L}
    isotropy = Isotropy(map)
    I = typeof(isotropy)
    new{D,I,T,L}(map)
  end
end

Base.size(a::TensorProductAffineMap{D}) where D = (D,)

function Base.getindex(a::TensorProductAffineMap{D},d::Integer) where D
  origin = Point(a.map.origin[d])
  gradient = diagonal_tensor(VectorValue(a.map.gradient[(d-1)*D+d]))
  AffineMap(gradient,origin)
end

get_factors(a::AffineMap) = TensorProductAffineMap(a)

function Isotropy(a::AffineMap{D,D}) where D
  o1 = a.origin[1]
  g1 = a.gradient[1]
  Isotropy(all(a.origin.data .== o1) && all([a.gradient[(d-1)*D+d] .== g1 for d=1:D]))
end

function Arrays.return_cache(a::AffineMap,x::AbstractTensorProductPoints{D}) where D
  factors = get_factors(a)
  points = get_factors(x)
  cache = return_cache(factors[1],points[1])
  r = _return_vec_cache(cache,D)
  return r,cache
end

function Arrays.evaluate!(_cache,a::AffineMap,x::AbstractTensorProductPoints{D}) where D
  r,cache = _cache
  factors = get_factors(a)
  points = get_factors(x)
  indices_map = get_indices_map(x)
  @inbounds for d = 1:D
    r[d] = evaluate!(cache,factors[d],points[d])
  end
  tensor_product_points(typeof(x),r,indices_map)
end

function Arrays.return_cache(a::AffineMap,x::AbstractTensorProductPoints{D,Isotropic}) where D
  factors = get_factors(a)
  points = get_factors(x)
  indices_map = get_indices_map(x)
  cache = return_cache(factors[1],points[1])
  return factors,indices_map,cache
end

function Arrays.evaluate!(_cache,a::AffineMap,x::AbstractTensorProductPoints{D,Isotropic}) where D
  factors,indices_map,cache = _cache
  points = get_factors(x)
  r = evaluate!(cache,factors[1],points[1])
  tensor_product_points(typeof(x),Fill(r,D),indices_map)
end

# gradients

Fields.gradient(a::TensorProductField) = GenericTensorProductField(get_factors(a))

function Arrays.evaluate!(
  cache,
  k::Broadcasting{typeof(∇)},
  a::AbstractArray{<:TensorProductField{D,I}}) where {D,I}

  factors = map(evaluate!,Fill(cache,D),Fill(k,D),get_factors.(a))
  GenericTensorProductField(factors,I())
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{typeof(∇∇)},
  a::AbstractArray{<:TensorProductField{D,I}}) where {D,I}

  factors = map(evaluate!,Fill(cache,D),Fill(k,D),get_factors.(a))
  GenericTensorProductField(factors,I())
end
