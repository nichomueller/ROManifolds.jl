"""
    struct AffineParamField{D1,D2,T,L} <: ParamField
      data::Vector{AffineField{D1,D2,T,L}}
    end

Parametric extension of a `AffineField` in `Gridap`
"""
struct AffineParamField{D1,D2,T,L} <: ParamField
  data::Vector{AffineField{D1,D2,T,L}}
end

function Fields.AffineField(
  gradient::ParamContainer{<:TensorValue},
  origin::ParamContainer{<:Point})

  data = map(AffineField,get_param_data(gradient),get_param_data(origin))
  AffineParamField(data)
end

get_param_data(f::AffineParamField) = f.data

function Arrays.return_cache(f::AffineParamField,x::Point)
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = parameterize(fix,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(param_getindex(f,i),x)
  end
  l,g
end

function Arrays.evaluate!(cache,f::AffineParamField,x::Point)
  l,g = cache
  @inbounds for i in param_eachindex(f)
    g[i] = evaluate!(l[i],param_getindex(f,i),x)
  end
  g
end

function Fields.gradient(f::AffineParamField)
  AffineParamField(map(gradient,f.data))
end

function Fields.push_∇∇(∇∇a::Field,ϕ::AffineParamField)
  Jt = ∇(ϕ)
  Jt_inv = pinvJt(Jt)
  Operation(Fields.push_∇∇)(∇∇a,Jt_inv)
end

function Arrays.lazy_map(
  k::Broadcasting{typeof(Fields.push_∇∇)},
  cell_∇∇a::AbstractArray,
  cell_map::AbstractArray{<:AffineParamField})
  cell_Jt = lazy_map(∇,cell_map)
  cell_invJt = lazy_map(Operation(pinvJt),cell_Jt)
  lazy_map(Broadcasting(Operation(Fields.push_∇∇)),cell_∇∇a,cell_invJt)
end

function Arrays.lazy_map(
  k::Broadcasting{typeof(Fields.push_∇∇)},
  cell_∇∇at::LazyArray{<:Fill{typeof(transpose)}},
  cell_map::AbstractArray{<:AffineParamField})
  cell_∇∇a = cell_∇∇at.args[1]
  cell_∇∇b = lazy_map(k,cell_∇∇a,cell_map)
  cell_∇∇bt = lazy_map(transpose,cell_∇∇b)
  cell_∇∇bt
end

function Fields.inverse_map(f::AffineParamField)
  AffineParamField(map(inverse_map,f.data))
end

function Base.zero(::Type{<:AffineParamField{D1,D2,T}}) where {D1,D2,T}
  @notimplemented "Must provide a parametric length"
end
