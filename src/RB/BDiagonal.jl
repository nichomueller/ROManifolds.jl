struct BDiagonal{T,MT,V}
  values::V
  size::Tuple{Int64,Int64}
  function BDiagonal(values::V) where {T,MT<:AbstractMatrix{T},V<:AbstractVector{MT}}
    new{T,MT,V}(values,mapreduce(size,(x,y)->x.+y,values))
  end
end

Base.getindex(b::BDiagonal{T,MT},j::Int64) where{T,MT} = b.values[j]
Base.size(b::BDiagonal{T,MT}) where {T,MT} = b.size
Base.size(b::BDiagonal{T,MT},i::Integer...) where {T,MT} = b.size[i...]

function Base.collect(b::BDiagonal{T,MT}) where{T,MT}
  (out,ri,ci) = zeros(T,size(b)),1,1
  for Dj in b.values
    (sz1,sz2) = size(Dj)
    @inbounds view(out,ri:(ri+sz1-1),ci:(ci+sz2-1)) .= Dj
    ri += sz1
    ci += sz2
  end
  out
end

## Basic linear algebra:
LinearAlgebra.factorize(b::BDiagonal{T,MT}) where{T,MT} = BDiagonal(factorize.(b.values))
LinearAlgebra.adjoint(b::BDiagonal{T,MT}) where{T,MT} = BDiagonal(adjoint.(b.values))

for (f,A) in Iterators.product((:(LinearAlgebra.ldiv!),:(LinearAlgebra.mul!)),(:StridedVector,:StridedMatrix))
  @eval begin
    function $f(target::$A{T},b::BDiagonal{T,MT},src::$A{T}) where{T,MT}
      ri = 1
      for Dj in b.values
        (sz1,sz2) = size(Dj)
        @inbounds $f(view(target,ri:(ri+sz1-1),:),Dj,view(src,ri:(ri+sz1-1),:))
        ri += sz1
      end
      target
    end
  end
end

function LinearAlgebra.:\(b::BDiagonal{T,MT},src::StridedArray{T}) where{T,MT}
  target = similar(src)
  MT <: Union{Factorization{T},Adjoint{T,<:Factorization{T}}} && return ldiv!(target,b,src)
  ldiv!(target,factorize(b),src)
end

Base.:*(b::BDiagonal{T,MT},src::StridedArray{T}) where{T,MT} = mul!(similar(src),b,src)

function Base.:*(src::Adjoint{T,<:StridedArray{T}},b::BDiagonal{T,MT}) where{T,MT}
  target = similar(src,T,(size(src,1),size(b,2)))
  ri = 1
  for Dj in b.values
    (sz1,sz2) = size(Dj)
    @inbounds mul!(view(target,:,ri:(ri+sz2-1)),view(src,:,ri:(ri+sz2-1)),Dj)
    ri += sz2
  end
  target
end
