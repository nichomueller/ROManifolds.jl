struct Contraction{X,Y} <: Function end

const ⊙₁₁ = Contraction{1,1}
const ⊙₁₂ = Contraction{1,2}
const ⊙₁₃ = Contraction{1,3}
const ⊙₂₁ = Contraction{2,1}
const ⊙₂₂ = Contraction{2,2}
const ⊙₂₂ = Contraction{2,2}
const ⊙₃₁ = Contraction{3,1}
const ⊙₃₂ = Contraction{3,2}
const ⊙₃₃ = Contraction{3,3}

function cmessage(::Contraction{X,Y},a::A,b::B) where {X,Y,A,B}
  error("Attempting to contract $A along axis $X with $B along axis $Y")
end

(c::AbstractContraction)(a::AbstractArray,b::AbstractArray) = contraction(c,a,b)

contraction(::Contraction,::AbstractArray,::AbstractArray) = @abstractmethod
contraction(c::Contraction,a::AbstractVector,b::AbstractVector) = cmessage(c,a,b)
contraction(c::Contraction,a::AbstractVector,b::AbstractMatrix) = cmessage(c,a,b)
contraction(c::Contraction,a::AbstractVector,b::AbstractArray{T,3}) where T = cmessage(c,a,b)
contraction(c::Contraction,a::AbstractMatrix,b::AbstractVector) = cmessage(c,a,b)
contraction(c::Contraction,a::AbstractMatrix,b::AbstractMatrix) = cmessage(c,a,b)
contraction(c::Contraction,a::AbstractMatrix,b::AbstractArray{T,3}) where T = cmessage(c,a,b)
contraction(c::Contraction,a::AbstractArray{T,3},b::AbstractVector) where T = cmessage(c,a,b)
contraction(c::Contraction,a::AbstractArray{T,3},b::AbstractMatrix) where T = cmessage(c,a,b)
contraction(c::Contraction,a::AbstractArray{T,3},b::AbstractArray{S,3}) where {T,S} = cmessage(c,a,b)

function contraction(::Contraction{1,1},a::AbstractVector,b::AbstractVector)
  dot(a,b)
end

function contraction(::Contraction{1,1},a::AbstractVector,b::AbstractMatrix)
  dropdims(a'*b;dims=1)
end

function contraction(::Contraction{1,2},a::AbstractVector,b::AbstractMatrix)
  b*a
end

function contraction(::Contraction{1,1},a::AbstractVector,b::AbstractArray{T,3}) where T
  dropdims(a'*b;dims=1)
end

const ⊙₃₄₁₂ = Contraction{(3,4),(1,2)}

function contraction(
  ::Contraction{(3,4),(1,2)},
  a::AbstractArray{T,4},
  b::AbstractArray{S,4}
  ) where {T,S}

end

const ⊙₄₅₆₁₂₃ = Contraction{(4,5,6),(1,2,3)}

function contraction(
  ::Contraction{(4,5,6),(1,2,3)},
  a::AbstractArray{T,6},
  b::AbstractArray{S,6}
  ) where {T,S}

end
