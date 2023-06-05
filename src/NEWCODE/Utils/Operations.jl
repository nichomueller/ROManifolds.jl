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

EMatrix(mats::Vector{Matrix{T}}) where T = EMatrix(hcat(mats...))

function allocate_matrix(::Type{Matrix{Float}},sizes...)
  Nr,Nc = sizes
  zeros(Nr,Nc)
end

function allocate_matrix(::Type{EMatrix{Float}},sizes...)
  Nr,Nc = sizes
  Elemental.zeros(EMatrix{Float},Nr,Nc)
end

complementary_dimension(mat::AbstractMatrix,ns::Int) = Int(size(mat,2)/ns)

function mode2_unfolding(mat::AbstractMatrix{Float},ns::Int)
  Ns,Nt = size(mat,1),complementary_dimension(mat,ns)
  mode2 = allocate_matrix(mat,Nt,Ns*ns)
  @inbounds for k = 1:ns
    setindex!(mode2,mat[:,(k-1)*Nt+1:k*Nt]',:,(k-1)*Ns+1:k*Ns)
  end
  return mode2
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

function get_findnz_vals(mat::Matrix{Float},findnz_idx::Vector{Int})
  selectdim(mat,1,findnz_idx)
end

function get_findnz_vals(mat::SparseMatrixCSC{Float,Int},findnz_idx::Vector{Int})
  mat[:][findnz_idx]
end

Base.getindex(emat::EMatrix{Float},::Colon,k::Int) = emat[:,k:k]

Base.getindex(emat::EMatrix{Float},k::Int,::Colon) = emat[k:k,:]

Base.getindex(emat::EMatrix{Float},idx::UnitRange{Int},k::Int) = emat[idx,k:k]

Base.getindex(emat::EMatrix{Float},k::Int,idx::UnitRange{Int}) = emat[k:k,idx]

# Base.:(-)(a::NTuple{N,T1},b::NTuple{N,T2}) where {N,T1,T2} = a.-b

# Base.:(*)(a::NTuple{N,T1},b::NTuple{N,T2}) where {N,T1,T2} = a.*b

# Base.adjoint(nt::NTuple{N,T}) where {N,T} = adjoint.(nt)

# Base.:(^)(vv::VectorValue,n::Int) = VectorValue([vv[k]^n for k=eachindex(vv)])

# Base.:(^)(vv::VectorValue,n::Float) = VectorValue([vv[k]^n for k=eachindex(vv)])

# Base.abs(vv::VectorValue) = VectorValue([abs(vv[k]) for k=eachindex(vv)])

Gridap.get_triangulation(m::Measure) = m.quad.trian
