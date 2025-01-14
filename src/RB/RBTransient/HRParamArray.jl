struct TupOfHRParamArray{T,N,A,B,C<:ParamArray{T,N},D}
  fe_quantity::A
  coeff::B
  hypred::NTuple{D,C}
end

function RBSteady.HRParamArray(
  fe_quantity::TupOfArrayContribution,
  coeff::TupOfArrayContribution,
  hypred::Tuple{Vararg{AbstractParamArray}})

  TupOfHRParamArray(fe_quantity,coeff,hypred)
end

const AbstractHRParamArray{T,N,A,B,C<:ParamArray{T,N}} = Union{
  HRParamArray{T,N,A,B,C},
  TupOfHRParamArray{T,N,A,B,C}
  }

Base.eltype(::TupOfHRParamArray{T}) where T = T
Base.eltype(::Type{<:TupOfHRParamArray{T}}) where T = T
Base.size(a::TupOfHRParamArray) = (length(a.hypred),)
Base.length(a::TupOfHRParamArray) = length(a.hypred)

for f in (:(Base.copy),:(Base.similar))
  @eval begin
    function $f(a::TupOfHRParamArray)
      fe_quantity′ = $f(a.fe_quantity)
      coeff′ = $f(a.coeff)
      hypred′ = map($f,a.hypred)
      HRParamArray(fe_quantity′,coeff′,hypred′)
    end
  end
end

function Base.fill!(a::TupOfHRParamArray,b::Number)
  map(h -> fill!(h,b),a.hypred)
end

function RBSteady.inv_project!(
  cache::HRParamArray,
  a::TupOfAffineContribution,
  b::TupOfArrayContribution)

  hypred = cache.hypred
  coeff = cache.coeff
  inv_project!(hypred,coeff,a,b)
end

function RBSteady.inv_project!(
  cache::TupOfHRParamArray,
  a::TupOfAffineContribution,
  b::TupOfArrayContribution)

  hypred = cache.hypred
  coeff = cache.coeff
  map(inv_project!,hypred,coeff,a,b)
end
