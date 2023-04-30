function LinearAlgebra.Vector(mat::Matrix{T}) where T
  Vector{T}(reshape(mat,:,))
end

function LinearAlgebra.Matrix(vec::Vector{T}) where T
  Matrix{T}(reshape(vec,:,1))
end

function LinearAlgebra.Matrix(block::Vector{<:AbstractArray{T}}) where T
  Matrix{T}(reduce(hcat,block))
end

function LinearAlgebra.Matrix(vblock::Vector{<:Vector{Vector{T}}}) where T
  n = length(vblock)
  mat = Matrix(vblock[1])
  if n > 1
    for i = 2:n
      mat = hcat(mat,Matrix(vblock[i]))
    end
  end
  mat
end

LinearAlgebra.Vector(dmat::EMatrix{T}) where T = Vector(Matrix(dmat))

LinearAlgebra.Matrix(dmat::EMatrix{T}) where T = convert(Matrix{T},dmat)

EMatrix(mat::Matrix{T}) where T = convert(EMatrix{T},mat)

function allocate_matrix(::Matrix{Float},sizes...)
  Nr,Nc = sizes
  zeros(Nr,Nc)
end

function allocate_matrix(::EMatrix{Float},sizes...)
  Nr,Nc = sizes
  Elemental.zeros(EMatrix{Float},Nr,Nc)
end

function blocks(mat::AbstractMatrix{Float};dims=(size(mat,1),1))
  @assert size(mat,1) == prod(dims)
  nblocks = size(mat,2)
  [reshape(mat[:,k],dims) for k = 1:nblocks]
end

function vblocks(mat::Matrix{T}) where T
  blockvec = Vector{T}[]
  for i in axes(mat,2)
    push!(blockvec,mat[:,i])
  end
  blockvec
end

function array3D(mat::AbstractMatrix,nrows::Int)
  ncols = Int(size(mat,1)/nrows)
  nblocks = size(mat,2)
  reshape(mat,nrows,ncols,nblocks)
end

function expand(tup::Tuple)
  ntup = ()
  for el = tup
    if typeof(el) <: Tuple
      ntup = (ntup...,expand(el)...)
    else
      ntup = (ntup...,el)
    end
  end
  ntup
end

function Base.NTuple(N::Int,T::DataType)
  NT = ()
  for _ = 1:N
    NT = (NT...,zero(T))
  end
  NT::NTuple{N,T}
end

function SparseArrays.findnz(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
  numnz = nnz(S)
  I = Vector{Ti}(undef, numnz)
  J = Vector{Ti}(undef, numnz)
  V = Vector{Tv}(undef, numnz)

  count = 1
  @inbounds for col = 1 : size(S, 2), k = SparseArrays.getcolptr(S)[col] : (SparseArrays.getcolptr(S)[col+1]-1)
      I[count] = rowvals(S)[k]
      J[count] = col
      V[count] = nonzeros(S)[k]
      count += 1
  end

  nz = findall(x -> x .!= 0., V)

  (I[nz], J[nz], V[nz])
end

function SparseArrays.findnz(x::SparseVector{Tv,Ti}) where {Tv,Ti}
  numnz = nnz(x)

  I = Vector{Ti}(undef, numnz)
  V = Vector{Tv}(undef, numnz)

  nzind = SparseArrays.nonzeroinds(x)
  nzval = nonzeros(x)

  @inbounds for i = 1 : numnz
      I[i] = nzind[i]
      V[i] = nzval[i]
  end

  nz = findall(v -> abs.(v) .>= eps(), V)

  (I[nz], V[nz])
end

function get_findnz_vals(arr::Matrix{Float},findnz_idx::Vector{Int})
  selectdim(arr,1,findnz_idx)
end

function get_findnz_vals(mat::SparseMatrixCSC{Float,Int},findnz_idx::Vector{Int})
  mat[:][findnz_idx]
end

Base.getindex(emat::EMatrix{Float},::Colon,k::Int) = emat[:,k:k]

Base.getindex(emat::EMatrix{Float},k::Int,::Colon) = emat[k:k,:]

Base.getindex(emat::EMatrix{Float},idx::UnitRange{Int},k::Int) = emat[idx,k:k]

Base.getindex(emat::EMatrix{Float},k::Int,idx::UnitRange{Int}) = emat[k:k,idx]

Gridap.VectorValue(D::Int,T::DataType) = VectorValue(NTuple(D,T))

function Base.one(vv::VectorValue{D,T}) where {D,T}
  vv_one = zero(vv) .+ one(T)
  vv_one::VectorValue{D,T}
end

function Base.Float64(vv::VectorValue{D,Float}) where D
  VectorValue(Float64.([vv...]))
end

function Base.Float32(vv::VectorValue{D,Float32}) where D
  VectorValue(Float32.([vv...]))
end

function Base.Int64(vv::VectorValue{D,Int}) where D
  VectorValue(Int64.([vv...]))
end

function Base.Int32(vv::VectorValue{D,Int32}) where D
  VectorValue(Int32.([vv...]))
end

function Gridap.evaluate(ntuple_fun::NTuple{N,Function},args...) where N
  tup = ()
  for fun in ntuple_fun
    tup = (tup...,fun(args...))
  end
  tup
end

Base.:(*)(a::Symbol,b::Symbol) = Symbol(String(a)*String(b))

Base.:(*)(a::Symbol,b::Symbol) = Symbol(String(a)*String(b))

Base.:(-)(a::NTuple{N,T1},b::NTuple{N,T2}) where {N,T1,T2} = a.-b

Base.:(*)(a::NTuple{N,T1},b::NTuple{N,T2}) where {N,T1,T2} = a.*b

Base.adjoint(nt::NTuple{N,T}) where {N,T} = adjoint.(nt)

Base.:(^)(vv::VectorValue,n::Int) = VectorValue([vv[k]^n for k=eachindex(vv)])

Base.:(^)(vv::VectorValue,n::Float) = VectorValue([vv[k]^n for k=eachindex(vv)])

Base.abs(vv::VectorValue) = VectorValue([abs(vv[k]) for k=eachindex(vv)])

Gridap.get_triangulation(m::Measure) = m.quad.trian

function Gridap.get_dirichlet_dof_values(f::MultiFieldFESpace)
  get_dirichlet_dof_values(first(f))
end
