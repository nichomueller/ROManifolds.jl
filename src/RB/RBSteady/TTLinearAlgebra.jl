struct TTContraction{T,N} <: AbstractArray{T,N}
  array::Array{T,N}
end

Base.size(a::TTContraction) = size(a.array)
Base.getindex(a::TTContraction{T,N},i::Vararg{Integer,N}) where {T,N} = getindex(a.array,i...)

function contraction(::AbstractArray...)
  @abstractmethod
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

function empirical_interpolation!(cache,C::AbstractArray{T,3}) where T
  @check size(C,1) == 1
  c...,Iv = cache
  A = dropdims(C;dims=1)
  I,Ai = empirical_interpolation!(c,A)
  push!(Iv,copy(I))
  return I,Ai
end

function _global_index(i,local_indices::Vector{Vector{Int32}})
  Iprev...,Ig = local_indices
  if length(Iprev) == 0
    return i
  end
  Il = last(Iprev)
  rankl = length(Il)
  islow = slow_index(i,rankl)
  ifast = fast_index(i,rankl)
  iprev = Il[ifast]
  giprev = _global_index(iprev,Iprev)
  return (giprev...,islow)
end

function _global_index(i,Il::Vector{Int32})
  rankl = length(Il)
  li = Il[fast_index(i,rankl)]
  gi = slow_index(i,rankl)
  return li,gi
end

function _to_split_global_indices(local_indices::Vector{Vector{Int32}},index_map::AbstractIndexMap)
  Is...,It = local_indices
  Igt = It
  Igs = copy(It)
  for (i,ii) in enumerate(Igt)
    ilsi,igti = _global_index(ii,last(Is))
    Igt[i] = igti
    Igs[i] = index_map[CartesianIndex(_global_index(ilsi,Is))]
  end
  return Igs,Igt
end

function _to_global_indices(local_indices::Vector{Vector{Int32}},index_map::AbstractIndexMap)
  if length(local_indices) != ndims(index_map) # this is the transient case
    @notimplementedif length(local_indices) != ndims(index_map)+1
    return _to_split_global_indices(local_indices,index_map)
  end
  Ig = local_indices[end]
  for (i,ii) in enumerate(Ig)
    Ig[i] = index_map[CartesianIndex(_global_index(ii,local_indices))]
  end
  return Ig
end

function _eim_cache(C::AbstractArray{T,3}) where T
  m,n = size(C,2),size(C,1)
  res = zeros(T,m)
  I = zeros(Int32,n)
  Iv = Vector{Int32}[]
  return C,I,res,Iv
end

function _next_core(Aprev::AbstractMatrix{T},Cnext::AbstractArray{T,3}) where T
  Cprev = reshape(Aprev,1,size(Aprev)...)
  _cores2basis(Cprev,Cnext)
end

function empirical_interpolation(index_map::AbstractIndexMap,cores::AbstractArray...)
  C,I,res,Iv = _eim_cache(first(cores))
  for i = eachindex(cores)
    _,Ai = empirical_interpolation!((I,res,Iv),C)
    if i < length(cores)
      C = _next_core(Ai,cores[i+1])
    else
      Ig = _to_global_indices(Iv,index_map)
      return Ig,Ai
    end
  end
end

function empirical_interpolation(index_map::SparseIndexMap,cores::AbstractArray...)
  empirical_interpolation(get_sparse_index_map(index_map),cores...)
end
