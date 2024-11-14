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
  basis::AbstractArray{T,3},
  coefficient::AbstractVector{S}
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

Base.@propagate_inbounds function contraction(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,3}
  ) where {T,S}

  @check size(factor1,2) == size(factor2,2)
  A = reshape(permutedims(factor1,(1,3,2)),:,size(factor1,2))
  B = reshape(permutedims(factor2,(2,1,3)),size(factor2,2),:)
  AB = A*B
  s1,s2,s3,s4 = size(factor1,1),size(factor1,3),size(factor2,1),size(factor2,3)
  ABp = permutedims(reshape(AB,s1,s2,s3,s4),(1,3,2,4))
  return TTContraction(ABp)
end

# product of cores on the component axis, for multivariate problems
Base.@propagate_inbounds function contraction(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,3},
  factor3::AbstractArray{U,3}
  ) where {T,S,U}

  ncomps1 = size(factor1,2)
  ncomps3 = size(factor3,2)
  cinds = CartesianIndices((ncomps1,ncomps3))
  @check size(factor2,2) == ncomps1*ncomps3
  A = reshape(permutedims(factor1,(2,1,3)),size(factor1,2),:)
  B = reshape(permutedims(factor2,(2,1,3)),size(factor2,2),:)
  C = reshape(permutedims(factor3,(2,1,3)),size(factor3,2),:)
  ABC = zeros(size(A,2),size(B,2),size(C,2))
  for (iA,a) = enumerate(eachcol(A))
    for (iB,b) = enumerate(eachcol(B))
      for (iC,c) = enumerate(eachcol(C))
        for (in,n) in enumerate(cinds)
          v = @views a[n.I[1]]*b[in]*c[n.I[2]]
          _entry!(+,ABC,v,iA,iB,iC)
        end
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
  factor1::AbstractArray{T,3},
  factor2::SparseCore{S,3},
  factor3::AbstractArray{U,3}
  ) where {T,S,U}

  sparsity = factor2.sparsity
  @check size(factor1,2) == DofMaps.num_rows(sparsity)
  @check DofMaps.num_cols(sparsity) == size(factor3,2)

  A = reshape(permutedims(factor1,(1,3,2)),:,size(factor1,2))
  B = reshape(permutedims(factor2,(1,3,2)),:,size(factor2,2))
  C = reshape(permutedims(factor3,(2,1,3)),size(factor3,2),:)
  BC = _sparsemul(B,C,sparsity)
  ABC = A*BC
  s1,s2 = size(factor1,1),size(factor1,3)
  s3,s4 = size(factor2,1),size(factor2,3)
  s5,s6 = size(factor3,1),size(factor3,3)
  ABCp = permutedims(reshape(ABC,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  return TTContraction(ABCp)
end

function sequential_product(::AbstractArray...)
  @abstractmethod
end

Base.@propagate_inbounds function sequential_product(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,3}
  ) where {T,S}

  @check size(factor1,3) == size(factor2,1)
  A = reshape(factor1,:,size(factor1,3))
  B = reshape(factor2,size(factor2,1),:)
  AB = A*B
  s1,s2,s3,s4 = size(factor1,1),size(factor1,2),size(factor2,2),size(factor2,3)
  reshape(AB,s1,s2*s3,s4)
end

Base.@propagate_inbounds function sequential_product(
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

Base.@propagate_inbounds function sequential_product(
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
  factor2,last_factors... = factors
  sequential_product(sequential_product(factor1,factor2),last_factors...)
end

function cores2basis(core::AbstractArray{Float64,3})
  reshape(core,:,size(core,3))
end

function cores2basis(cores::AbstractArray{Float64,3}...)
  core = sequential_product(cores...)
  dropdims(core;dims=1)
end

function cores2basis(dof_map::AbstractDofMap{D},cores::AbstractArray{Float64,3}...) where D
  @check length(cores) ≥ D
  coresD = sequential_product(cores[1:D]...)
  invmap = invert(dof_map)
  coresD′ = view(coresD,:,vec(invmap),:)
  if length(cores) == D
    dropdims(coresD′;dims=1)
  else
    cores2basis(coresD′,cores[D+1:end]...)
  end
end

function galerkin_projection(
  cores_test::Vector{<:AbstractArray{Float64,3}},
  cores::Vector{<:AbstractArray{Float64,3}})

  rcores = contraction.(cores_test,cores)
  rcore = sequential_product(rcores...)
  dropdims(rcore;dims=(1,2))
end

function galerkin_projection(
  cores_test::Vector{<:AbstractArray{Float64,3}},
  cores::Vector{<:AbstractArray{Float64,3}},
  cores_trial::Vector{<:AbstractArray{Float64,3}})

  rcores = contraction.(cores_test,cores,cores_trial)
  rcore = sequential_product(rcores...)
  dropdims(rcore;dims=(1,2,3))
end

function reduced_coupling(
  cores_p::Vector{<:AbstractArray{Float64,3}},
  B::Rank1Tensor,
  cores_d::Vector{<:AbstractArray{Float64,3}})

  factors = get_factors(B)
  @check length(cores_p)-1 == length(factors) == length(cores_d)
  @check all(size(f,2) == size(c,2) for (f,c) in zip(factors,cores))
  rcores = contraction.(cores_p,factors,cores_d)
  rcore = sequential_product(rcores...)
  dropdims(rcore;dims=(1,2,3))
end

# utils

Base.@propagate_inbounds function _sparsemul(B,C,sparsity::SparsityPatternCSC)
  BC = zeros(size(B,1),DofMaps.num_rows(sparsity),size(C,2))
  rv = rowvals(sparsity)
  for (iB,b) in enumerate(eachrow(B))
    for (iC,c) in enumerate(eachcol(C))
      for (irow,ci) in enumerate(c)
        for nzi in nzrange(sparsity,irow)
          _entry!(+,BC,b[nzi]*ci,iB,rv[nzi],iC)
        end
      end
    end
  end
  return reshape(permutedims(BC,(2,1,3)),size(BC,2),:)
end

@inline function _entry!(combine::Function,A::AbstractArray{T,3},v,i,j,k) where T
  aijk = A[i,j,k]
  A[i,j,k] = combine(aijk,v)
  A
end

# empirical interpolation

function basis_index(i,cores_indices::Vector{Vector{Ti}}) where Ti
  Iprevs...,Icurr = cores_indices
  if length(Iprevs) == 0
    return i
  end
  Iprevprevs...,Iprev = Iprevs
  rankprev = length(Iprev)
  icurr = slow_index(i,rankprev)
  iprevs = basis_index(Iprev[fast_index(i,rankprev)],Iprevs)
  return (iprevs...,icurr)
end

function basis_indices(
  ::Val{1},
  cores_indices::Vector{Vector{Ti}},
  dof_map::AbstractDofMap{D}
  )::Vector{Ti} where {Ti,D}

  Iprev...,Icurr = cores_indices
  basis_indices = zeros(Ti,length(Icurr))
  for (k,ik) in enumerate(Icurr)
    indices_k = basis_index(ik,cores_indices)
    basis_indices[k] = dof_map[CartesianIndex(indices_k[1:D])]
  end
  return basis_indices
end

function basis_indices(
  ::Val{N},
  cores_indices::Vector{Vector{Ti}},
  dof_map::AbstractDofMap{D}
  )::Vector{Vector{Ti}} where {Ti,D,N}

  L = length(cores_indices)

  Iprev...,Icurr = cores_indices
  basis_indices = zeros(Ti,length(Icurr),N)
  for (k,ik) in enumerate(Icurr)
    indices_k = basis_index(ik,cores_indices)
    basis_indices[k,1] = dof_map[CartesianIndex(indices_k[1:D])]
    for (il,l) in enumerate(D+1:L)
      basis_indices[k,1+il] = indices_k[l]
    end
  end

  return collect.(eachcol(basis_indices))
end

function basis_indices(cores_indices::Vector{<:Vector},dof_map::AbstractDofMap{D}) where D
  L = length(cores_indices)
  ninds = L - D + 1
  return basis_indices(Val(ninds),cores_indices,dof_map)
end

function basis_indices(cores_indices::Vector{<:Vector},dof_map::SparseDofMap)
  basis_indices(cores_indices,get_sparse_dof_map(dof_map))
end
