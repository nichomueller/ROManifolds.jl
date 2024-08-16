abstract type PartialFunctions <: Function end

struct PartialDerivative{N} <: PartialFunctions end

PartialDerivative{N}(f) where N = Operation(PartialTrace{N}())(∇(f))

const ∂ₓ₁ = PartialDerivative{1}
const ∂ₓ₂ = PartialDerivative{2}
const ∂ₓ₃ = PartialDerivative{3}

function Arrays.evaluate!(cache,::Broadcasting{<:PartialDerivative{N}},f) where N
  Broadcasting(Operation(PartialTrace{N}()))(Broadcasting(∇)(f))
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
