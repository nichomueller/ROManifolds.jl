function RBSteady.contraction(
  factor1::Array{T,3},
  factor2::Array{S,3},
  factor3::Array{U,3},
  combine::Function
  ) where {T,S,U}

  @check size(factor1,2) == size(factor2,2) == size(factor3,2)
  A = reshape(permutedims(factor1,(2,1,3)),size(factor1,2),:)
  B = reshape(permutedims(factor2,(2,1,3)),size(factor2,2),:)
  C = reshape(permutedims(factor3,(2,1,3)),size(factor3,2),:)
  ABC = zeros(size(A,2),size(B,2),size(C,2))
  ABC′ = zeros(size(A,2),size(B,2),size(C,2))
  cache = zeros(size(factor1,2))
  cache′ = zeros(size(factor1,2)-1)
  @inbounds for (iA,a) = enumerate(eachcol(A))
    for (iB,b) = enumerate(eachcol(B))
      cache .= a .* b
      cache′ .= a[2:end] .* b[2:end]
      for (iC,c) = enumerate(eachcol(C))
        ABC[iA,iB,iC] = sum(cache .* c)
        ABC′[iA,iB,iC] = sum(cache′ .* c[1:end-1])
      end
    end
  end
  s1,s2 = size(factor1,1),size(factor1,3)
  s3,s4 = size(factor2,1),size(factor2,3)
  s5,s6 = size(factor3,1),size(factor3,3)
  ABCp = permutedims(reshape(ABC,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  ABCp′ = permutedims(reshape(ABC′,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  return TTContraction(combine(ABCp,ABCp′))
end
