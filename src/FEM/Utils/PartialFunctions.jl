"""
    abstract type PartialFunctions <: Map end
"""
abstract type PartialFunctions <: Map end

"""
    struct PartialDerivative{N} <: PartialFunctions end

Map that implements a partial derivative in `Gridap`
"""
struct PartialDerivative{N} <: PartialFunctions end

PartialDerivative{N}(f) where N = Operation(PartialTrace{N}())(∇(f))

const ∂₁ = PartialDerivative{1}
const ∂₂ = PartialDerivative{2}
const ∂₃ = PartialDerivative{3}

const ∂₁₂ = PartialDerivative{(1,2)}
const ∂₁₂₃ = PartialDerivative{(1,2,3)}
const ∂ₙ = Union{∂₁₂,∂₁₂₃}

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
