abstract type PartialFunctions <: Function end

struct PartialDerivative{N} <: PartialFunctions end

PartialDerivative{N}(f) where N = Operation(PartialTrace{N}())(∇(f))

const ∂ₓ₁ = PartialDerivative{1}
const ∂ₓ₂ = PartialDerivative{2}
const ∂ₓ₃ = PartialDerivative{3}

function Arrays.evaluate!(cache,::Broadcasting{<:PartialDerivative{N}},f) where N
  Broadcasting(Operation(PartialTrace{N}()))(Broadcasting(∇)(f))
end

function PartialDerivative{N}(f::TProductCellField) where N
  g = GenericTProductCellField(PartialDerivative{1}.(f.single_fields),f.trian)
  return GenericTProductDiffCellField(PartialDerivative{N}(),f,g)
end

function PartialDerivative{N}(f::TProductFEBasis) where N
  dbasis = PartialDerivative{1}.(f.basis)
  g = GenericTProductCellField(dbasis,f.trian)
  return GenericTProductDiffCellField(PartialDerivative{N}(),f,g)
end

const PartialDerivativeTProductCellField{N,A,B,C} = GenericTProductDiffCellField{PartialDerivative{N},A,B,C}

function Arrays.return_cache(k::Operation{typeof(*)},α::TProductCellDatum,β::PartialDerivativeTProductCellField)
  cache = return_cache(k,α,get_data(β))
  diff_cache = return_cache(k,α,TProduct.get_diff_data(β))
  return cache,diff_cache
end

function Arrays.return_cache(k::Operation{typeof(*)},α::PartialDerivativeTProductCellField,β::TProductCellDatum)
  cache = return_cache(k,get_data(α),β)
  diff_cache = return_cache(k,α,TProduct.get_diff_data(β),β)
  return cache,diff_cache
end

function Arrays.evaluate!(_cache,k::Operation{typeof(*)},α::TProductCellDatum,β::PartialDerivativeTProductCellField{N}) where N
  cache,diff_cache = _cache
  αβ = evaluate!(cache,k,α,get_data(β))
  dαβ = evaluate!(cache,k,α,TProduct.get_diff_data(β))
  return GenericTProductDiffCellField(PartialDerivative{N},αβ,dαβ)
end

function Arrays.evaluate!(_cache,k::Operation{typeof(*)},α::PartialDerivativeTProductCellField{N},β::TProductCellDatum) where N
  cache,diff_cache = _cache
  αβ = evaluate!(cache,k,get_data(α),β)
  dαβ = evaluate!(cache,k,TProduct.get_diff_data(α),β)
  return GenericTProductDiffCellField(PartialDerivative{N},αβ,dαβ)
end

function TProduct.tproduct_array(
  ::Type{<:PartialDerivative{N}},
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  index_map,
  args...) where N

  TProduct.tp_sort!(arrays_1d,index_map)
  TProduct.tp_sort!(gradients_1d,index_map)
  TProductPDerivativeArray{N}(arrays_1d,gradients_1d)
end

function PartialDerivative{N}(f::Function,x::Point,fx) where N
  PartialTrace{N}(gradient(f,x,fx))
end

struct PartialTrace{N} <: PartialFunctions end

(f::PartialTrace{N})(x...) where N = evaluate(f,x...)
(f::PartialTrace{N})(a::CellField) where N = Operation(f)(a)
(f::PartialTrace{N})(a::Field...) where N = Operation(f)(a...)

(f::PartialTrace{N})(v::MultiValue) where N = PartialTrace{N}(v)
PartialTrace{N}(v::MultiValue) where N = @notimplemented
PartialTrace{N}(v::MultiValue{Tuple{D}}) where {N,D} = v[N]
