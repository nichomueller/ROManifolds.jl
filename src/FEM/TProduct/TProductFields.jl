struct TensorProductMap{D,T} <: Map
  values::NTuple{D,T}
end

function Arrays.return_value(
  ::TensorProductMap{D},
  a::NTuple{D,T}
  ) where {D,T<:AbstractArray{<:Number}}

  ka = kronecker(a...) |> collect
  return ka
end

function Arrays.return_cache(
  ::TensorProductMap{D},
  a::NTuple{D,T}
  ) where {D,T<:AbstractArray{<:Number}}

  ka = kronecker(a...) |> collect
  return ka
end

function Arrays.evaluate!(
  cache,
  ::TensorProductMap{D},
  a::NTuple{D,T}
  ) where {D,T<:AbstractArray{<:Number}}

  ka = kronecker(a...)
  copyto!(cache,ka)
  return ka
end

function Arrays.return_value(
  ::TensorProductMap{D},
  a::NTuple{D,T},
  x::Point
  ) where {D,T<:Union{Field,AbstractArray{Field}}}

  factors = ntuple(d->return_value(a[d],x),D)
  lazy_kron = kronecker(factors...)
  collect(lazy_kron)
end

_get_product_cache(t::NTuple{N,<:AbstractArray} where N) = collect(kronecker(t...))
_get_product_cache(t::NTuple{N,<:Tuple} where N) = _get_product_cache(first.(t))

function Arrays.return_cache(
  m::TensorProductMap{D},
  a::NTuple{D,T},
  x::Point
  ) where {D,T<:Union{Field,AbstractArray{Field}}}

  factors = ntuple(d->return_cache(a[d],x),D)
  V = return_type(m,a,x)
  array_factors = Vector{V}(undef,D)
  product = _get_product_cache(factors)
  return factors,array_factors,product
end

function Arrays.evaluate!(
  cache,
  ::TensorProductMap{D},
  a::NTuple{D,T},
  x::Point
  ) where {D,T<:Union{Field,AbstractArray{Field}}}

  factors,array_factors,product = cache
  @inbounds for d = 1:D
    array_factors[d] = evaluate!(factors[d],a[d],x)
  end
  copyto!(product,kronecker(array_factors...))
  return product
end

abstract type TensorProductField <: Field end

struct GenericTensorProductField{D,F} <: Field
  fields::NTuple{D,F}
end

function Arrays.return_value(f::GenericTensorProductField,x::Point)
  return_value(TensorProductMap(f.fields),x)
end

for T in (:(Point),:(AbstractVector{<:Point}))
  @eval begin
    function Arrays.return_cache(f::GenericTensorProductField,x::$T)
      return_cache(TensorProductMap(f.fields),x)
    end
    function Arrays.evaluate!(cache,f::GenericTensorProductField,x::$T)
      evaluate!(cache,TensorProductMap(f.fields),x)
    end
  end
end

function Arrays.return_value(f::GenericTensorProductField,x::TensorProductNodes)
  TensorProductArray(ntuple(d->return_value(f[d],x.nodes[d]),D))
end

function Arrays.return_cache(f::GenericTensorProductField,x::TensorProductNodes)
  TensorProductArray(ntuple(d->return_cache(f[d],x.nodes[d]),D))
end

function Arrays.evaluate!(cache,f::GenericTensorProductField,x::TensorProductNodes)
  TensorProductArray(ntuple(d->evaluate!(cache[d],f[d],x.nodes[d]),D))
end
