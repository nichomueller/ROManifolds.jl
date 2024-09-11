function IndexMaps.recast(i::SparseIndexMap,a::AbstractVector{<:AbstractArray{T,3}}) where T
  us = IndexMaps.get_univariate_sparsity(i)
  @check length(us) ≤ length(a)
  if length(us) == length(a)
    return map(SparseCore,a,us)
  else
    asparse = map(i ->SparseCore(a[i],us[i]),eachindex(us))
    afull = a[length(us)+1:end]
    return [asparse...,afull...]
  end
end

"""
    abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

Type for nonstandard representations of tensor train cores.

Subtypes:
- [`SparseCore`](@ref)

"""
abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

"""
    abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

Tensor train cores for sparse matrices.

Subtypes:
- [`SparseCoreCSC`](@ref)

"""
abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

function _cores2basis(a::SparseCore{S},b::SparseCore{T}) where {S,T}
  @notimplemented "Need to provide a sparse index map for the construction of the global basis"
end

"""
    struct SparseCoreCSC{T,Ti} <: SparseCore{T,3} end

Tensor train cores for sparse matrices in CSC format

"""
struct SparseCoreCSC{T,Ti} <: SparseCore{T,3}
  array::Array{T,3}
  sparsity::SparsityPatternCSC{T,Ti}
end

function SparseCore(array::Array{T,3},sparsity::SparsityPatternCSC{T}) where T
  SparseCoreCSC(array,sparsity)
end

Base.size(a::SparseCoreCSC) = size(a.array)
Base.getindex(a::SparseCoreCSC,i::Vararg{Integer,3}) = getindex(a.array,i...)

num_space_dofs(a::SparseCoreCSC) = IndexMaps.num_rows(a.sparsity)*IndexMaps.num_cols(a.sparsity)

# block cores

struct BlockCore{T,D,A<:AbstractArray{T,D},BS} <: AbstractArray{T,D}
  array::Vector{A}
  axes::BS
  function BlockCore(array::Vector{A},axes::BS) where {T,D,A<:AbstractArray{T,D},BS<:NTuple}
    @assert all((size(a,2)==size(first(array),2) for a in array))
    new{T,D,A,BS}(array,axes)
  end
end

function BlockCore(array::Vector{<:AbstractArray},touched::AbstractArray{Bool}=I(length(array)))
  block_sizes = _sizes_from_blocks(array,touched)
  axes = map(blockedrange,block_sizes)
  BlockCore(array,axes)
end

function BlockCore(array::Vector{<:Vector{<:AbstractArray}})
  N = length(array)
  map(eachindex(first(array))) do n
    touched = n == 1 ? fill(true,N) : I(N)
    arrays_n = getindex.(array,n)
    BlockCore(arrays_n,touched)
  end
end

const BlockCore3D{T} = BlockCore{T,3,Array{T,3}}

Base.size(a::BlockCore) = map(length,axes(a))
Base.axes(a::BlockCore) = a.axes

function Base.getindex(a::BlockCore3D,i::Vararg{Integer,3})
  if size(a,1) == 1
    _getindex_vector(a,i...)
  else
    _getindex_matrix(a,i...)
  end
end

function _getindex_vector(a::BlockCore3D,i::Vararg{Integer,3})
  i1,i2,i3 = i
  @assert i1 == 1
  b3 = BlockArrays.findblockindex(axes(a,3),i3)
  a.array[b3.I...][i1,i2,b3.α...]
end

function _getindex_matrix(a::BlockCore3D,i::Vararg{Integer,3})
  i1,i2,i3 = i
  b1 = BlockArrays.findblockindex(axes(a,1),i1)
  b3 = BlockArrays.findblockindex(axes(a,3),i3)
  if b1.I == b3.I
    a.array[b1.I...][b1.α...,i2,b3.α...]
  else
    zero(eltype(a))
  end
end

function _sizes_from_blocks(a::Vector{<:AbstractArray},touched::AbstractVector{Bool})
  s1 = fill(1,length(a))
  s2 = fill(size(a[1],2),length(a))
  s3 = map(a -> size(a,3),a)
  for i in 1:length(a)-1
    s1[i] = 0
    s2[i] = 0
  end
  return (s1,s2,s3)
end

function _sizes_from_blocks(a::Vector{<:AbstractArray},touched::AbstractMatrix{Bool})
  s1 = map(a -> size(a,1),a)
  s2 = fill(size(a[1],2),length(a))
  s3 = map(a -> size(a,3),a)
  for i in 1:length(a)-1
    s2[i] = 0
  end
  return (s1,s2,s3)
end

# core operations

function cat_cores(a::AbstractArray{T,3},b::AbstractArray{T,3}) where T
  @check size(a,2) == size(b,2)
  if size(a,1) == size(b,1)
    ab = cat(a,b;dims=3)
  else
    ab = similar(a,size(a,1)+size(b,1),size(a,2),size(a,3)+size(b,3))
    fill!(ab,zero(T))
    @views ab[axes(a,1),:,axes(a,3)] = a
    @views ab[size(a,1)+1:end,:,size(a,3)+1:end] = b
  end
  return ab
end

function cat_cores(a::AbstractArray{T,3},b::AbstractMatrix{T}) where T
  @check size(a,2) == size(b,2)
  ab = similar(a,size(a,1)+size(b,1),size(a,2),size(a,3)+1)
  fill!(ab,zero(T))
  @views ab[axes(a,1),:,axes(a,3)] = a
  @views ab[size(a,1)+1:end,:,end] = b
  return ab
end

function pushlast(a::AbstractArray{T,3},b::AbstractMatrix{T}) where T
  @check size(a,2) == size(b,2)
  s1,s2,s3 = size(a)
  s1′ = size(b,1)
  ab = similar(a,s1,s2,s3+1)
  fill!(ab,zero(T))
  @views ab[:,:,1:s3] = a
  @views ab[s1-s1′+1:end,:,s3+1] = b
  return ab
end

function Base.:*(a::AbstractMatrix{T},b::AbstractArray{S,3}) where {T,S}
  @check size(a,2) == size(b,2)
  TS = promote_type(T,S)
  ab = zeros(TS,size(b,1),size(a,1),size(b,3))
  @inbounds for i = axes(b,1), j = axes(b,3)
    ab[i,:,j] = a*b[i,:,j]
  end
  return ab
end

function Base.:*(a::AbstractArray{T,3},b::AbstractMatrix{S}) where {T,S}
  @check size(a,2) == size(b,1)
  TS = promote_type(T,S)
  ab = zeros(TS,size(a,1),size(b,2),size(a,3))
  @inbounds for i = axes(a,1), j = axes(a,3)
    ab[i,:,j] = a[i,:,j]*b
  end
  return ab
end

"""
    cores2basis(index_map::AbstractIndexMap,cores::AbstractArray...) -> AbstractMatrix
    cores2basis(index_map::AbstractIndexMap,cores::ArrayBlock...) -> ArrayBlock

Computes the kronecker product of the suitably indexed input cores

"""
function cores2basis(index_map::AbstractIndexMap,cores::AbstractArray...)
  cores2basis(_cores2basis(index_map,cores...))
end

function cores2basis(cores::AbstractArray...)
  c2m = _cores2basis(cores...)
  return dropdims(c2m;dims=1)
end

function cores2basis(core::AbstractArray{T,3}) where T
  pcore = permutedims(core,(2,1,3))
  return reshape(pcore,size(pcore,1),:)
end

function _cores2basis(a::AbstractArray{S,3},b::AbstractArray{T,3}) where {S,T}
  @check size(a,3) == size(b,1)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ab = zeros(TS,size(a,1),nrows,size(b,3))
  for i = axes(a,1), j = axes(b,3)
    for α = axes(a,3)
      @inbounds cache = ab[i,:,j]
      _kronadd!(cache,b[α,:,j],a[i,:,α])
      @inbounds ab[i,:,j] = cache
    end
  end
  return ab
end

# when we multiply a 4-D spatial core with a 3-D temporal core
function _cores2basis(a::AbstractArray{S,4},b::AbstractArray{T,3}) where {S,T}
  @check size(a,4) == size(b,1)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ncols = size(a,3)
  ab = zeros(TS,size(a,1),nrows*ncols,size(b,3)) # returns a 3-D array
  for i = axes(a,1), j = axes(b,3)
    for α = axes(a,4)
      @inbounds cache = ab[i,:,j]
      _kronadd!(cache,b[α,:,j],a[i,:,:,α])
      @inbounds ab[i,:,j] = cache
    end
  end
  return ab
end

function _cores2basis(a::AbstractArray{S,3},b::AbstractArray{T,4}) where {S,T}
  @notimplemented "Usually the spatial cores are computed before the temporal ones"
end

function _cores2basis(a::AbstractArray{S,N},b::AbstractArray{T,N}) where {S,T,N}
  @abstractmethod
end

function _cores2basis(a::AbstractArray,b::AbstractArray...)
  c,d... = b
  return _cores2basis(_cores2basis(a,c),d...)
end

function _cores2basis(i::AbstractIndexMap,a::AbstractArray{T,3}...) where T
  basis = _cores2basis(a...)
  invi = inv_index_map(i)
  return view(basis,:,vec(invi),:)
end

# when we multiply two SparseCoreCSC objects, the result is a 3-D core that stacks
# the matrices' rows and columns
function _cores2basis(
  I::SparseIndexMap,
  a::SparseCoreCSC{S},
  b::SparseCoreCSC{T}
  ) where {S,T}

  @check size(a,3) == size(b,1)
  Is = get_sparse_index_map(I)
  TS = promote_type(T,S)
  ndofs = num_space_dofs(a)*num_space_dofs(b)
  ab = zeros(TS,size(a,1),ndofs,size(b,3))
  _sparse_cores2basis!(ab,Is,a,b)
  return ab
end

function _cores2basis(
  I::SparseIndexMap,
  a::SparseCoreCSC{S},
  b::SparseCoreCSC{T},
  c::SparseCoreCSC{U}
  ) where {S,T,U}

  @check size(a,3) == size(b,1) && size(b,3) == size(c,1)
  Is = get_sparse_index_map(I)
  TSU = promote_type(T,S,U)
  ndofs = num_space_dofs(a)*num_space_dofs(b)*num_space_dofs(c)
  abc = zeros(TSU,size(a,1),ndofs,size(c,3))
  _sparse_cores2basis!(abc,Is,a,b,c)
  return abc
end

function _sparse_cores2basis!(ab,I::AbstractIndexMap,a,b)
  vI = vec(I)
  for i = axes(a,1), j = axes(b,3)
    for α = axes(a,3)
      @inbounds cache = ab[i,vI,j]
      _kronadd!(cache,b[α,:,j],a[i,:,α])
      @inbounds ab[i,vI,j] = cache
    end
  end
  return ab
end

function _sparse_cores2basis!(abc,I::AbstractIndexMap,a,b,c)
  vI = vec(I)
  for i = axes(a,1), j = axes(c,3)
    for α = axes(a,3), β = axes(b,3)
      @inbounds cache = ab[i,vI,j]
      _kronadd!(cache,c[β,:,j],b[α,:,β],a[i,:,α])
      @inbounds abc[i,vI,j] = cache
    end
  end
  return abc
end

# Fixed dofs

function _sparse_cores2basis!(ab,I::FixedDofsIndexMap,a,b)
  nz_indices = findall(I[:].!=0)
  for i = axes(a,1), j = axes(b,3)
    for α = axes(a,e)
      @inbounds @views ab[i,vec(I),j] += kron(b[α,:,j],a[i,:,α]
        )[nz_indices]
    end
  end
  return ab
end

function _sparse_cores2basis!(abc,I::FixedDofsIndexMap,a,b,c)
  nz_indices = findall(I[:].!=0)
  for i = axes(a,1), j = axes(c,3)
    for α = axes(a,3), β = axes(b,3)
      @inbounds @views abc[i,vec(I),j] += kron(c[β,:,j],b[α,:,β],a[i,:,α]
        )[nz_indices]
    end
  end
  return abc
end

# core compression

function compress_core(a::AbstractArray{T,3},btest::AbstractArray{S,3}) where {T,S}
  TS = promote_type(T,S)
  ra_prev,ra = size(a,1),size(a,3)
  rV_prev,rV = size(btest,1),size(btest,3)
  ab = zeros(TS,rV_prev,ra_prev,rV,ra)
  for ib1 = 1:rV_prev
    @inbounds b′ = btest[ib1,:,:]
    for ia1 = 1:ra_prev
      @inbounds a′ = a[ia1,:,:]
      for ib3 = 1:rV
        @inbounds b′′ = b′[:,ib3]
        for ia3 = 1:ra
          @inbounds a′′ = a′[:,ia3]
          @inbounds ab[ib1,ia1,ib3,ia3] = dot(b′′,a′′)
        end
      end
    end
  end
  return ab
end

function compress_core(
  a::AbstractArray{T,3},
  btrial::AbstractArray{S,3},
  btest::AbstractArray{S,3}
  ) where {T,S}

  TS = promote_type(T,S)
  ra_prev,ra = size(a,1),size(a,3)
  rU_prev,rU = size(btrial,1),size(btrial,3)
  rV_prev,rV = size(btest,1),size(btest,3)
  bab = zeros(TS,rV_prev,ra_prev,rU_prev,rV,ra,rU)
  w = zeros(TS,size(a,2))
  for ibU1 = 1:rU_prev
    @inbounds bU′ = btrial[ibU1,:,:]
    for ia1 = 1:ra_prev
      @inbounds a′ = a[ia1,:,:]
      for ibU3 = 1:rU
        @inbounds bU′′ = bU′[:,ibU3]
        for ia3 = 1:ra
          @inbounds a′′ = a′[:,ia3]
          w .= a′′.*bU′′
          for ibV1 = 1:rV_prev
            @inbounds bV′ = btest[ibV1,:,:]
            for ibV3 = 1:rV
              @inbounds bV′′ = bV′[:,ibV3]
              @inbounds bab[ibV1,ia1,ibU1,ibV3,ia3,ibU3] = dot(bV′′,w)
            end
          end
        end
      end
    end
  end
  return bab
end

function compress_core(
  a::SparseCore{T},
  btrial::AbstractArray{S,3},
  btest::AbstractArray{S,3}
  ) where {T,S}

  TS = promote_type(T,S)
  ra_prev,ra = size(a,1),size(a,3)
  rU_prev,rU = size(btrial,1),size(btrial,3)
  rV_prev,rV = size(btest,1),size(btest,3)
  bab = zeros(TS,rV_prev,ra_prev,rU_prev,rV,ra,rU)
  w = zeros(TS,IndexMaps.num_rows(a.sparsity))
  for ibU1 = 1:rU_prev
    @inbounds bU′ = btrial[ibU1,:,:]
    for ia1 = 1:ra_prev
      @inbounds a′ = a[ia1,:,:]
      for ibU3 = 1:rU
        @inbounds bU′′ = bU′[:,ibU3]
        for ia3 = 1:ra
          @inbounds a′′ = a′[:,ia3]
          _sparsemul!(w,a′′,bU′′,a.sparsity)
          for ibV1 = 1:rV_prev
            @inbounds bV′ = btest[ibV1,:,:]
            for ibV3 = 1:rV
              @inbounds bV′′ = bV′[:,ibV3]
              @inbounds bab[ibV1,ia1,ibU1,ibV3,ia3,ibU3] = dot(bV′′,w)
            end
          end
        end
      end
    end
  end
  return bab
end

function multiply_cores(a::AbstractArray{T,4},b::AbstractArray{S,4}) where {T,S}
  @check (size(a,3)==size(b,1) && size(a,4)==size(b,2))
  TS = promote_type(T,S)
  ra1,ra2 = size(a,1),size(a,2)
  rb3,rb4 = size(b,3),size(b,4)
  ab = zeros(TS,ra1,ra2,rb3,rb4)

  for ia1 = 1:ra1
    @inbounds a′ = a[ia1,:,:,:]
    for ia2 = 1:ra2
      @inbounds a′′ = a′[ia2,:,:]
      for ib3 = 1:rb3
        @inbounds b′ = b[:,:,ib3,:]
        for ib4 = 1:rb4
          @inbounds b′′ = b′[:,:,ib4]
          @inbounds ab[ia1,ia2,ib3,ib4] = dot(a′′,b′′)
        end
      end
    end
  end

  return ab
end

function multiply_cores(a::AbstractArray{T,6},b::AbstractArray{S,6}) where {T,S}
  @check (size(a,4)==size(b,1) && size(a,5)==size(b,2) && size(a,6)==size(b,3))
  TS = promote_type(T,S)
  ra1,ra2,ra3 = size(a,1),size(a,2),size(a,3)
  rb4,rb5,rb6 = size(b,4),size(b,5),size(b,6)
  ab = zeros(TS,ra1,ra2,ra3,rb4,rb5,rb6)

  for ia1 = 1:ra1
    @inbounds a′ = a[ia1,:,:,:,:,:]
    for ia2 = 1:ra2
      @inbounds a′′ = a′[ia2,:,:,:,:]
      for ia3 = 1:ra3
        @inbounds a′′′ = a′′[ia3,:,:,:]
        for ib4 = 1:rb4
          @inbounds b′ = b[:,:,:,ib4,:,:]
          for ib5 = 1:rb5
            @inbounds b′′ = b′[:,:,:,ib5,:]
            for ib6 = 1:rb6
              @inbounds b′′′ = b′′[:,:,:,ib6]
              @inbounds ab[ia1,ia2,ia3,ib4,ib5,ib6] = dot(a′′′,b′′′)
            end
          end
        end
      end
    end
  end

  return ab
end

function multiply_cores(c1::AbstractArray,cores::AbstractArray...)
  _c1,_cores... = cores
  multiply_cores(multiply_cores(c1,_c1),_cores...)
end

function _dropdims(a::AbstractArray{T,4}) where T
  @check size(a,1) == size(a,2) == 1
  dropdims(a;dims=(1,2))
end

function _dropdims(a::AbstractArray{T,6}) where T
  @check size(a,1) == size(a,2) == size(a,3) == 1
  dropdims(a;dims=(1,2,3))
end

function _kronadd!(c::AbstractVector,a::AbstractVector,b::AbstractVector)
  @check length(c) == length(a)*length(b)
  rb = length(b)
  for i in eachindex(c)
    @inbounds c[i] += a[cld(i,rb)]*b[(i-1)%rb+1]
  end
end

function _kronadd!(c::AbstractMatrix,a::AbstractMatrix,b::AbstractMatrix)
  @check size(c,1) == size(a,1)*size(b,1)
  @check size(c,2) == size(a,2)*size(b,2)
  rb,cb = size(b)
  for i in axes(c,1), j in axes(c,2)
    @inbounds c[i,j] += a[cld(i,rb),cld(j,cb)]*b[(i-1)%rb+1,(j-1)%cb+1]
  end
end

function _kronadd!(c::AbstractMatrix,a::AbstractMatrix,b::AbstractVector)
  @check size(c,1) == size(a,1)*size(b,1)
  @check size(c,2) == size(a,2)
  rb = length(b)
  for i in axes(c,1)
    bi = b[(i-1)%rb+1]
    for j in axes(c,2)
      @inbounds c[i,j] += a[cld(i,rb),j]*bi
    end
  end
end

function _kronadd!(c::AbstractMatrix,a::AbstractVector,b::AbstractMatrix)
  @check size(c,1) == size(a,1)*size(b,1)
  @check size(c,2) == size(b,2)
  rb,cb = size(b)
  for i in axes(c,1)
    ai = a[cld(i,rb)]
    for j in axes(c,2)
      @inbounds c[i,j] += ai*b[(i-1)%rb+1,(j-1)%cb+1]
    end
  end
end

function _sparsemul!(c::AbstractVector,nzv::AbstractVector,b::AbstractVector,sparsity::SparsityPatternCSC)
  rv = rowvals(sparsity)
  fill!(c,zero(eltype(c)))
  @inbounds for icol in eachindex(b)
    bi = b[icol]
    for irow in nzrange(sparsity,icol)
      c[rv[irow]] += nzv[irow]*bi
    end
  end
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
