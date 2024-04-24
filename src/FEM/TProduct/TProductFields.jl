abstract type TensorProductField{D,I} <: AbstractVector{Field} end

get_factors(::TensorProductField) = @abstractmethod
get_field(::TensorProductField) = @abstractmethod
get_isotropy(::TensorProductField{D,I} where D) where I = I

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

struct GenericTensorProductField{D,I,A,B} <: TensorProductField{D,I}
  factors::A
  field::B
  isotropy::I
  function GenericTensorProductField(factors::A,field::B,isotropy::I=Isotropy(factors)) where {A,B,I}
    D = length(factors)
    new{D,I,A,B}(factors,field,isotropy)
  end
end

Base.size(a::GenericTensorProductField) = size(a.field)
Base.getindex(a::GenericTensorProductField,i::Integer...) = getindex(a.field,i...)

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

struct TensorProductAffineMap{D,T,L} <: AbstractVector{AffineMap{1,1,T,1}}
  map::AffineMap{D,D,T,L}
end

Base.size(a::TensorProductAffineMap{D}) where D = (D,)

function Base.getindex(a::TensorProductAffineMap{D},d::Integer) where D
  origin = Point(a.map.origin[d])
  gradient = diagonal_tensor(VectorValue(a.map.gradient[(d-1)*D+d]))
  AffineMap(gradient,origin)
end

get_factors(a::AffineMap) = TensorProductAffineMap(a)

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

Fields.gradient(a::TensorProductField) = GenericTensorProductField(get_factors(a),get_field(a))

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
