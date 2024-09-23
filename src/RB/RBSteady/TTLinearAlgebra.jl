struct TTContraction{T,N} <: AbstractArray{T,N}
  array::Array{T,N}
end

Base.size(a::TTContraction) = size(a.array)
Base.getindex(a::TTContraction{T,N},i::Vararg{Integer,N}) where {T,N} = getindex(a.array,i...)

function contraction(::AbstractArray...)
  @abstractmethod
end

function contraction!(::Union{AbstractArray,Number}...)
  @abstractmethod
end

function contraction(
  basis::Array{T,3},
  coefficient::Vector{S}
  ) where {T,S}

  @check size(basis,2) == length(coefficient)
  A = reshape(permutedims(basis,(1,3,2)),:,size(basis,2))
  v = A*coefficient
  s1,s2 = size(basis,1),size(basis,3)
  M = reshape(v,s1,s2)
  return M
end

function contraction!(
  cache::AbstractMatrix,
  basis::AbstractArray{T,3} where T,
  coefficient::AbstractVector,
  α::Number=1,β::Number=0
  )

  @check size(cache) == (size(basis,1),size(basis,3))
  @check size(basis,2) == length(coefficient)
  v = view(cache,:)
  A = reshape(permutedims(basis,(1,3,2)),:,size(basis,2))
  mul!(v,A,coefficient,α,β)
  return
end

function contraction!(
  cache::AbstractArray{U,3} where U,
  basis::AbstractArray{T,3} where T,
  coefficient::AbstractMatrix,
  α::Number=1,β::Number=0
  )

  @check size(cache,3) == size(coefficient,2)
  for (i,c) in enumerate(eachslice(cache,dims=3))
    @views coeff = coefficient[:,i]
    contraction!(c,basis,coeff,α,β)
  end
end

function contraction(
  factor1::Array{T,3},
  factor2::Array{S,3}
  ) where {T,S}

  @check size(factor1,2) == size(factor2,2)
  A = reshape(permutedims(factor1,(1,3,2)),:,size(factor1,2))
  B = reshape(permutedims(factor2,(2,1,3)),size(factor2,2),:)
  AB = A*B
  s1,s2,s3,s4 = size(factor1,1),size(factor1,3),size(factor2,1),size(factor2,3)
  ABp = permutedims(reshape(AB,s1,s2,s3,s4),(1,3,2,4))
  return TTContraction(ABp)
end

function contraction(
  factor1::Array{T,3},
  factor2::Array{S,3},
  factor3::Array{U,3}
  ) where {T,S,U}

  @check size(factor1,2) == size(factor2,2) == size(factor3,2)
  A = reshape(permutedims(factor1,(1,3,2)),:,size(factor1,2))
  B = reshape(permutedims(factor2,(1,3,2)),:,size(factor2,2))
  C = reshape(permutedims(factor3,(1,3,2)),:,size(factor3,2))
  ABC = zeros(size(A,2),size(B,2),size(C,2))
  @inbounds for (iA,a) = enumerate(eachcol(A))
    for (iB,b) = enumerate(eachcol(B))
      for (iC,c) = enumerate(eachcol(C))
        ABC[iA,iB,iC] = sum(a .* b .* c)
      end
    end
  end
  s1,s2 = size(factor1,1),size(factor1,3)
  s3,s4 = size(factor2,1),size(factor2,3)
  s5,s6 = size(factor3,1),size(factor3,3)
  ABCp = permutedims(reshape(ABC,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  return TTContraction(ABCp)
end

function contraction(
  factor1::Array{T,3},
  factor2::SparseCore{S,3},
  factor3::Array{U,3}
  ) where {T,S,U}

  sparsity = factor2.sparsity
  @check size(factor1,2) == IndexMaps.num_rows(sparsity)
  @check IndexMaps.num_cols(sparsity) == size(factor3,2)

  A = reshape(permutedims(factor1,(1,3,2)),:,size(factor1,2))
  B = reshape(permutedims(factor2,(2,1,3)),size(factor2,2),:)
  C = reshape(permutedims(factor3,(2,1,3)),size(factor3,2),:)
  AB = _sparsemul(A,B,sparsity)
  ABC = zeros(size(A,2),size(B,2),size(C,2))
  @inbounds for (iA,a) = enumerate(eachcol(A))
    for (iB,b) = enumerate(eachcol(B))
      for (iC,c) = enumerate(eachcol(C))
        ABC[iA,iB,iC] = sum(a .* b .* c)
      end
    end
  end
  s1,s2 = size(factor1,1),size(factor1,3)
  s3,s4 = size(factor2,1),size(factor2,3)
  s5,s6 = size(factor3,1),size(factor3,3)
  ABCp = permutedims(reshape(ABC,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  return TTContraction(ABCp)
end

function _sparsemul(A,B,sparsity::SparsityPatternCSC)
  AB = zeros(size(A,1),size(B,2),size(A,2))
  rv = rowvals(sparsity)
  for (iB,b) in enumerate(eachcol(B))
    for (iA,a) in enumerate(eachrow(A))
      for (irow,ai) in enumerate(a)
        for nzi in nzrange(sparsity,irow)
          AB[iA,rv[nzi],iB] += ai*b[nzi]
        end
      end
    end
  end
  return AB
end

function sequential_product(::AbstractArray...)
  @abstractmethod
end

function sequential_product(
  factor1::Array{T,3},
  factor2::Array{S,3}
  ) where {T,S}

  @check size(factor1,3) == size(factor2,1)
  A = reshape(factor1,:,size(factor1,3))
  B = reshape(factor2,size(factor2,1),:)
  AB = A*B
  s1,s2,s3,s4 = size(factor1,1),size(factor1,2),size(factor2,2),size(factor2,3)
  reshape(AB,s1,s2*s3,s4)
end

function sequential_product(
  factor1::AbstractArray{T,4},
  factor2::TTContraction{S,4}
  ) where {T,S}

  @check size(factor1,1) == size(factor1,2) == 1
  @check size(factor1,3) == size(factor2,1)
  @check size(factor1,4) == size(factor2,2)

  a = vec(factor1)
  B = reshape(factor2,length(a),:)
  aB = a'*B
  s1,s2,s3,s4 = 1,1,size(factor2,3),size(factor2,4)
  reshape(aB,s1,s2,s3,s4)
end

function sequential_product(
  factor1::AbstractArray{T,6},
  factor2::TTContraction{S,6}
  ) where {T,S}

  @check size(factor1,1) == size(factor1,2) == size(factor1,3) == 1
  @check size(factor1,4) == size(factor2,1)
  @check size(factor1,5) == size(factor2,2)
  @check size(factor1,6) == size(factor2,3)

  a = vec(factor1)
  B = reshape(factor2,length(a),:)
  aB = a'*B
  s1,s2,s3,s4,s5,s6 = 1,1,1,size(factor2,4),size(factor2,5),size(factor2,6)
  reshape(aB,s1,s2,s3,s4,s5,s6)
end

function sequential_product(factor1::AbstractArray,factors::AbstractArray...)
  factor2,last_factors... = b
  sequential_product(sequential_product(factor1,factor2),last_factors...)
end

function cores2basis(cores::Vector{<:AbstractArray{Float64,3}})
  core = sequential_product(cores...)
  dropdims(core;dims=1)
end

function reduced_cores(
  cores_test::Vector{<:AbstractArray{Float64,3}},
  cores::Vector{<:AbstractArray{Float64,3}})

  rcores = contraction.(cores_test,cores)
  rcore = sequential_product(rcores...)
  dropdims(rcore;dims=(1,2))
end

function reduced_cores(
  cores_test::Vector{<:AbstractArray{Float64,3}},
  cores::Vector{<:AbstractArray{Float64,3}},
  cores_trial::Vector{<:AbstractArray{Float64,3}})

  rcores = contraction.(cores_test,cores,cores_trial)
  rcore = sequential_product(rcores...)
  dropdims(rcore;dims=(1,2,3))
end

# empirical interpolation

function basis_index(i,cores_indices::Vector{Vector{Int32}})
  Iprevs...,Icurr = cores_indices
  if length(Iprev) == 0
    return i
  end
  Iprevprevs...,Iprev = Iprev
  rankprev = length(Iprev)
  icurr = slow_index(i,rankprev)
  iprevs = basis_index(Iprev[fast_index(i,rankprev)],Iprevs)
  return (giprev...,icurr)
end

function basis_indices(cores_indices::Vector{Vector{Int32}},index_map::AbstractIndexMap{D}) where D
  L = length(cores_indices)
  ninds = L - D + 1

  Iprev...,Icurr = cores_indices
  basis_indices = fill(copy(Icurr),ninds)
  for (k,ik) in enumerate(Icurr)
    indices_k = basis_index(ik,Iprev)
    basis_indices[1][k] = index_map[CartesianIndex(indices_k[1:D])]
    for l = D+1:L
      basis_indices[l][k] = indices_k[l]
    end
  end

  return basis_indices
end

function basis_indices(cores_indices::Vector{Vector{Int32}},index_map::SparseIndexMap)
  basis_indices(cores_indices,get_sparse_index_map(index_map))
end
