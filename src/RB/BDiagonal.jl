struct BDiagonal{T,MT}
  sz::Tuple{Int64,Int64}
  V::Vector{MT}
end

BDiagonal(V) = BDiagonal{eltype(first(V)),eltype(V)}(mapreduce(size,(x,y)->x.+y,V),V)

Base.getindex(BD::BDiagonal{T,MT},j::Int64) where{T,MT} = BD.V[j]
Base.size(BD::BDiagonal{T,MT}) where {T,MT} = BD.sz

function Base.collect(BD::BDiagonal{T,MT}) where{T,MT}
  (out,ri,ci) = zeros(T,size(BD)),1,1
  for Dj in BD.V
    (sz1,sz2) = size(Dj)
    @inbounds view(out,ri:(ri+sz1-1),ci:(ci+sz2-1)) .= Dj
    ri += sz1
    ci += sz2
  end
  out
end

## Basic linear algebra:
LinearAlgebra.factorize(BD::BDiagonal{T,MT}) where{T,MT} = BDiagonal(factorize.(BD.V))
LinearAlgebra.adjoint(BD::BDiagonal{T,MT}) where{T,MT} = BDiagonal(adjoint.(BD.V))

for (fn,ar) in Iterators.product((:ldiv!,:mul!),(:StridedVector,:StridedMatrix))
  @eval begin
    function $fn(target::$ar{T},BD::BDiagonal{T,MT},src::$ar{T}) where{T,MT}
      ri = 1
      for Dj in BD.V
        (sz1,sz2) = size(Dj)
        @inbounds $fn(view(target,ri:(ri+sz1-1),:),Dj,view(src,ri:(ri+sz1-1),:))
        ri += sz1
      end
      target
    end
  end
end

function LinearAlgebra.:\(BD::BDiagonal{T,MT},src::StridedArray{T}) where{T,MT}
  target = similar(src)
  MT <: Union{Factorization{T},Adjoint{T,<:Factorization{T}}} && return ldiv!(target,BD,src)
  ldiv!(target,factorize(BD),src)
end

Base.:*(BD::BDiagonal{T,MT},src::StridedArray{T}) where{T,MT} = mul!(similar(src),BD,src)
