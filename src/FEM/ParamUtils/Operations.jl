abstract type PartialFunctions <: Function end

struct PartialDerivative{N,F} <: PartialFunctions
  f::F
  PartialDerivative{N}(f::F) where {N,F} = new{N,F}(f)
end

PartialDerivative{1}(f) = Operation(PartialTrace{1}())(∇(f))
PartialDerivative{2}(f) = Operation(PartialTrace{2}())(∇(f))
PartialDerivative{3}(f) = Operation(PartialTrace{3}())(∇(f))

const ∂ₓ₁{F} = PartialDerivative{1,F}
const ∂ₓ₂{F} = PartialDerivative{2,F}
const ∂ₓ₃{F} = PartialDerivative{3,F}

function Arrays.evaluate!(cache,::Broadcasting{<:PartialDerivative{N}},f) where N
  Broadcasting(Operation(PartialTrace{N}()))(Broadcasting(∇)(f))
end

struct PartialTrace{N} <: PartialFunctions end

(f::PartialTrace{N})(x...) where N = evaluate(f,x...)
(f::PartialTrace{N})(a::CellField) where N = Operation(f)(a)
(f::PartialTrace{N})(a::Field...) where N = Operation(f)(a...)

function PartialDerivative(f::Function,x::Point,fx)
  PartialTrace{N}(gradient(f,x,fx))
end

(f::PartialTrace{N})(v::MultiValue) where N = PartialTrace{N}(v)
PartialTrace{N}(v::MultiValue) where N = @notimplemented
PartialTrace{N}(v::MultiValue{Tuple{D}}) where {N,D} = v[N]
