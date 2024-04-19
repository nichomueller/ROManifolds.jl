abstract type TensorProductField{D,I} <: Field end

get_factors(::TensorProductField) = @abstractmethod
get_field(::TensorProductField) = @abstractmethod
get_isotropy(::TensorProductField{D,I} where D) where I = I

Fields.gradient(a::TensorProductField) = GenericTensorProductField(map(gradient,get_factors(a)))

function Arrays.evaluate!(
  cache,
  k::Broadcasting{typeof(∇)},
  a::AbstractArray{<:TensorProductField{D,I}}) where {D,I}

  TensorProductField(map(evaluate!,Fill(cache,D),Fill(k,D),get_factors.(a)))
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{typeof(∇∇)},
  a::AbstractArray{<:TensorProductField{D,I}}) where {D,I}

  TensorProductField(map(evaluate!,Fill(cache,D),Fill(k,D),get_factors.(a)))
end

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
  function GenericTensorProductField(factors::A,isotropy::I) where {A,I}
    D = length(factors)
    new{D,I,A}(isotropy,factors)
  end
end

function GenericTensorProductField(factors::A) where A
  isotropy = Isotropy(all(factors .== factors[1]))
  GenericTensorProductField(factors,isotropy)
end

get_factors(a::GenericTensorProductField) = a.factors
get_field(a::GenericTensorProductField) = GenericField(x->sum(map(f->evaluate.(f,x),a.factors)))

function Arrays.return_cache(a::GenericTensorProductField,x::AbstractTensorProductPoints)
  factors = get_factors(a)
  points = get_factors(x)
  c = return_cache(factors[1],points[1])
  r = Vector{typeof(get_array(c))}(undef,D)
  return r,c
end

function Arrays.evaluate!(cache,a::GenericTensorProductField,x::AbstractTensorProductPoints{D}) where D
  r,c = cache
  factors = get_factors(a)
  points = get_factors(x)
  index_map = get_index_map(x)
  @inbounds for d = 1:D
    r[d] = evaluate!(c,factors[d],points[d])
  end
  tpr = FieldFactors(r,index_map,Anisotropic())
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
  index_map = get_index_map(x)
  r = evaluate!(cache,factors[1],points[1])
  tpr = FieldFactors(Fill(r,D),index_map,Isotropic())
  return tpr
end

# gradients

function Arrays.evaluate!(
  cache,
  k::Broadcasting{typeof(∇)},
  a::GenericTensorProductField{D,I,FieldGradientArray{N}}
  ) where {D,I,N}

  ∇a = map(evaluate!,Fill(cache,D),Fill(k,D),map(get_factors,a))
  GenericTensorProductField(∇a)
end
