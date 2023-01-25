function Base.Matrix(v::Vector{T}) where T
  Matrix{T}(reshape(v,:,1))
end

function Base.Matrix(vblock::Vector{Vector{T}}) where T
  Matrix{T}(reduce(vcat,transpose.(vblock))')
end

function Base.Matrix(mblock::Vector{T}) where {T<:AbstractMatrix}
  @assert check_dimensions(mblock) "Wrong dimensions"
  T(reduce(hcat,mblock))
end

function Base.Matrix(vblock::Vector{Vector{Vector{T}}}) where T
  n = length(vblock)
  mat = Matrix(vblock[1])
  if n > 1
    for i = 2:n
      mat = hcat(mat,Matrix(vblock[i]))
    end
  end
  mat
end

const Block{T} = Vector{T} where {T<:AbstractArray{Ts} where Ts}
const BlockMatrix{T} = Block{Matrix{T}} where T

function blocks(mat::Matrix{T},nrow::Int) where T
  blocks(mat,size(mat,2);dims=(nrow,:))
end

function blocks(mat::Matrix{T},nblocks=size(mat,2);dims=(size(mat,1),1)) where T
  @assert check_dimensions(mat,nblocks) "Wrong dimensions"

  ncol_block = Int(size(mat)[2]/nblocks)
  idx2 = ncol_block:ncol_block:size(mat)[2]
  idx1 = idx2 .- ncol_block .+ 1

  blockmat = Matrix{T}[]
  for i in eachindex(idx1)
    block_i = Matrix(reshape(mat[:,idx1[i]:idx2[i]],dims))
    push!(blockmat,block_i)
  end
  blockmat
end

function blocks(mat::NTuple{N,Matrix{T}},args...;kwargs...) where {N,T}
  Broadcasting(m->blocks(m,args...;kwargs...))(mat)
end

function blocks(mat::Array{T,3};dims=size(mat)[1:2]) where T
  blockmat = Matrix{T}[]
  for nb = 1:size(mat,3)
    block_i = Matrix(reshape(mat[:,:,nb],dims))
    push!(blockmat,block_i)
  end
  blockmat
end

function vblocks(vec::Vector{T}) where T
  [vec]
end

function vblocks(mat::Matrix{T}) where T
  blockvec = Vector{T}[]
  for i in axes(mat,2)
    push!(blockvec,mat[:,i])
  end
  blockvec
end

check_dimensions(vb::AbstractVector) =
  all([size(vb[i])[1] == size(vb[1])[1] for i = 2:length(vb)])
check_dimensions(m::AbstractMatrix,nb::Int) = iszero(size(m)[2]%nb)

spacetime_vector(mat::AbstractMatrix) = mat[:]
spacetime_vector(fun::Function) = u -> fun(u)[:]

istuple(tup::Any) = false
istuple(tup::Tuple) = true

function expand(tup::Tuple)
  ntup = ()
  for el = tup
    if istuple(el)
      ntup = (ntup...,expand(el)...)
    else
      ntup = (ntup...,el)
    end
  end
  ntup
end

function SparseArrays.sparsevec(M::Matrix{T},findnz_map::Vector{Int}) where T
  sparse_vblocks = SparseVector{T}[]
  for j = axes(M,2)
    push!(sparse_vblocks,sparsevec(findnz_map,M[:,j],maximum(findnz_map)))
  end

  sparse_vblocks
end

function sparsevec_to_sparsemat(svec::SparseVector,Nc::Int)
  ij,v = findnz(svec)
  i,j = from_vec_to_mat_idx(ij,Nc)
  sparse(i,j,v,maximum(i),Nc)
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

function remove_zero_rows(mat::AbstractMatrix;tol=eps())
  sum_cols = sum(mat,dims=2)
  nzrows = findall(x -> abs(x) ≥ tol,sum_cols)
  mat[nzrows,:]
end

function LinearAlgebra.kron(
  mat1::AbstractArray,
  mat2::NTuple{N,AbstractArray}) where N
  Broadcasting(m2->kron(mat1,m2))(mat2)
end

function LinearAlgebra.kron(
  mat1::NTuple{N,AbstractArray},
  mat2::AbstractArray) where N
  Broadcasting(m1->kron(m1,mat2))(mat1)
end

function LinearAlgebra.kron(
  mat1::NTuple{N,AbstractArray},
  mat2::NTuple{N,AbstractArray}) where N
  kron.(mat1,mat2)
end

function LinearAlgebra.kron(
  b1::BlockMatrix,
  b2::BlockMatrix)

  n1 = length(b1)
  n2 = length(b2)
  [kron(b1[i],b2[j]) for i=1:n1 for j=1:n2]
end

function Base.NTuple(N::Int,T::DataType)
  NT = ()
  for _ = 1:N
    NT = (NT...,zero(T))
  end
  NT::NTuple{N,T}
end

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
