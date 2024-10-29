eltype2(x) = eltype(eltype(x))

function Algebra.allocate_vector(::Type{V},n::Integer) where V<:AbstractParamVector
  @warn "Allocating a vector of unit parametric length, will likely result in an error"
  vector = allocate_vector(eltype(V),n)
  consecutive_param_array(vector,1)
end

function Algebra.allocate_vector(::PType{V,L},n::Integer) where {V<:AbstractParamVector,L}
  vector = allocate_vector(eltype(V),n)
  consecutive_param_array(vector,L)
end

function Algebra.allocate_vector(::Type{<:BlockParamVector{T}},indices::BlockedUnitRange) where T
  V = ConsecutiveParamVector{T}
  mortar(map(ids -> allocate_vector(V,ids),blocks(indices)))
end

function Algebra.allocate_vector(::PType{<:BlockParamVector{T},L},indices::BlockedUnitRange) where {T,L}
  V = ConsecutiveParamVector{T}
  PV = ParamType{V,L}
  mortar(map(ids -> allocate_vector(PV,ids),blocks(indices)))
end

function Algebra.allocate_in_range(::Type{V},matrix::AbstractParamMatrix) where V<:AbstractParamVector
  rows = ParamDataStructures.inneraxes(matrix)[1]
  L = param_length(matrix)
  PV = ParamType{V,L}
  allocate_vector(PV,rows)
end

function Algebra.allocate_in_range(matrix::AbstractParamMatrix{T}) where T
  V = ConsecutiveParamVector{T}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_range(matrix::BlockParamMatrix{T}) where T
  V = BlockConsecutiveParamVector{T}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_domain(::Type{V},matrix::AbstractParamMatrix) where V<:AbstractParamVector
  cols = ParamDataStructures.inneraxes(matrix)[2]
  L = param_length(matrix)
  PV = ParamType{V,L}
  allocate_vector(PV,cols)
end

function Algebra.allocate_in_domain(matrix::AbstractParamMatrix{T}) where T
  V = ConsecutiveParamVector{T}
  allocate_in_domain(V,matrix)
end

function Algebra.allocate_in_domain(matrix::BlockParamMatrix{T}) where T
  V = BlockConsecutiveParamVector{T}
  allocate_in_domain(V,matrix)
end

@inline function Algebra._add_entries!(
  combine::Function,A,vs::AbstractParamMatrix,is,js)

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          vij = get_param_entry(vs,li,lj)
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A,vs::AbstractParamVector,is)

  for (li,i) in enumerate(is)
    if i>0
      vi = get_param_entry(vs,li)
      add_entry!(combine,A,vi,i)
    end
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamVector,v::Number,i)
  @inbounds for k = param_eachindex(A)
    aik = A[k][i]
    A[k][i] = combine(aik,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamVector,v::AbstractVector,i)
  @inbounds for k = param_eachindex(A)
    aik = A[k][i]
    vk = v[k]
    A[k][i] = combine(aik,vk)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamVector,v::Number,i)
  data = get_all_data(A)
  @inbounds for k = param_eachindex(A)
    aik = data[i,k]
    data[i,k] = combine(aik,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamVector,v::AbstractVector,i)
  data = get_all_data(A)
  @inbounds for k = param_eachindex(A)
    aik = data[i,k]
    vk = v[k]
    data[i,k] = combine(aik,vk)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::Number,i,j)
  @inbounds for k = param_eachindex(A)
    aijk = A[k][i,j]
    A[k][i,j] = combine(aijk,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::AbstractVector,i,j)
  @inbounds for k = param_eachindex(A)
    aijk = A[k][i,j]
    vk = v[k]
    A[k][i,j] = combine(aijk,vk)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamSparseMatrix,v::Number,i,j)
  l = nz_index(A,i,j)
  nz = get_all_data(nonzeros(A))
  @inbounds for k = param_eachindex(A)
    aijk = nz[l,k]
    nz[l,k] = combine(aijk,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamSparseMatrix,v::AbstractVector,i,j)
  l = nz_index(A,i,j)
  nz = get_all_data(nonzeros(A))
  @inbounds for k = param_eachindex(A)
    aijk = nz[l,k]
    vk = v[k]
    nz[l,k] = combine(aijk,vk)
  end
  A
end

function Algebra.is_entry_stored(::PType{T},i,j) where T
  is_entry_stored(eltype(T),i,j)
end

# sparse functionalities

function Algebra.allocate_coo_vectors(::PType{T,L},n::Integer) where {Tv,Ti,T<:ParamSparseMatrix{Tv,Ti},L}
  I = zeros(Ti,n)
  J = zeros(Ti,n)
  V = zeros(Tv,n)
  PV = consecutive_param_array(V,L)
  I,J,PV
end

@inline function Algebra.push_coo!(::PType{<:ParamSparseMatrix},I,J,V,i,j,v)
  @notimplemented "Cannot push to ParamArray"
end

"""
    ParamCounter{C}

Extends the concept of `counter` in Gridap to accommodate a parametric setting.

"""
struct ParamCounter{C}
  counter::C
  plength::Int
end

ParamDataStructures.param_length(a::ParamCounter) = a.plength

Algebra.LoopStyle(::Type{<:ParamCounter{C}}) where C = LoopStyle(C)

@inline function Algebra.add_entry!(::typeof(+),a::ParamCounter,v,i,j)
  add_entry!(+,a.counter,v,i,j)
end

function Algebra.nz_counter(builder::SparseMatrixBuilder{<:PType{T,L}},axes) where {T,L}
  Tv = eltype(T)
  counter = nz_counter(SparseMatrixBuilder(Tv),axes)
  ParamCounter(counter,L)
end

function Algebra.nz_allocation(a::Algebra.ArrayCounter{<:PType{T,L}}) where {T,L}
  Tv = eltype(T)
  v = similar(Tv,map(length,a.axes))
  fill!(v,zero(eltype(v)))
  consecutive_param_array(v,L)
end

# csc

function Algebra.sparse_from_coo(::PType{<:ParamSparseMatrixCSC},I,J,V,m,n)
  sparse(I,J,V,m,n)
end

@inline function Algebra.is_entry_stored(::PType{<:ParamSparseMatrixCSC},i,j)
  true
end

function Algebra.finalize_coo!(::PType{<:ParamSparseMatrixCSC},I,J,V,m,n)
  nothing
end

Base.@propagate_inbounds function Algebra.nz_index(A::ParamSparseMatrixCSC,i0::Integer,i1::Integer)
  if !(1 <= i0 <= innersize(A,1) && 1 <= i1 <= innersize(A,2)); throw(BoundsError()); end
  ptrs = SparseArrays.getcolptr(A)
  r1 = Int(ptrs[i1])
  r2 = Int(ptrs[i1+1]-1)
  (r1 > r2) && return -1
  r1 = searchsortedfirst(rowvals(A),i0,r1,r2,Base.Order.Forward)
  ((r1 > r2) || (rowvals(A)[r1] != i0)) ? -1 : r1
end

# alternative implementation
# We assumes same sparsity across parameters, to be generalized in the future

function Algebra.nz_allocation(a::ParamCounter{<:Algebra.CounterCSC{Tv,Ti}}) where {Tv,Ti}
  counter = a.counter
  colptr = Vector{Ti}(undef,counter.ncols+1)
  @inbounds for i in 1:counter.ncols
    colptr[i+1] = counter.colnnzmax[i]
  end
  length_to_ptrs!(colptr)
  plength = a.plength
  ndata = colptr[end] - one(Ti)
  pndata = ndata*plength
  rowval = Vector{Ti}(undef,ndata)
  nzval = zeros(Tv,pndata)
  colnnz = counter.colnnzmax
  fill!(colnnz,zero(Ti))
  ParamInserterCSC(counter.nrows,counter.ncols,colptr,colnnz,rowval,nzval,plength)
end

"""
    struct ParamInserterCSC{Tv,Ti} end

Extends the concept of `InserterCSC` in Gridap to accommodate a parametric setting.
Tv is the type of the parametric nonzero entries of the CSC matrices to be
assembled.

"""
struct ParamInserterCSC{Tv,Ti}
  nrows::Int
  ncols::Int
  colptr::Vector{Ti}
  colnnz::Vector{Ti}
  rowval::Vector{Ti}
  nzval::Vector{Tv}
  plength::Int
end

ParamDataStructures.param_length(inserter::ParamInserterCSC) = inserter.plength

Algebra.LoopStyle(::Type{<:ParamInserterCSC}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC,v::Nothing,i,j)
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check  pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
  end
  nothing
end

@noinline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC,v::Number,i,j)
  add_entry!(+,a,fill(v,param_length(a)),i,j)
end

@noinline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC,v::AbstractArray,i,j)
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  ndata = length(a.rowval)
  pndata = length(a.nzval)
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for (l,vl) in enumerate(p:ndata:pndata)
      a.nzval[vl] = v[l]
    end
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
      @inbounds for l in k:ndata:pndata
        a.nzval[l+1] = a.nzval[l]
      end
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for (l,vl) in enumerate(p:ndata:pndata)
      a.nzval[vl] = v[l]
    end
  else
    # update existing entry
    @inbounds for (l,vl) in enumerate(p:ndata:pndata)
      a.nzval[vl] += v[l]
    end
  end
  nothing
end

function Algebra.create_from_nz(a::ParamInserterCSC)
  k = 1
  ndata = a.colptr[end]-1
  pndata = length(a.nzval)
  plength = param_length(a)
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      @inbounds for (il,l) in enumerate(p:ndata:pndata)
        α = k + (il-1)*ndata
        a.nzval[α] = a.nzval[l]
      end
      a.rowval[k] = a.rowval[p]
      k += 1
    end
  end
  @inbounds for j in 1:a.ncols
    a.colptr[j+1] = a.colnnz[j]
  end
  length_to_ptrs!(a.colptr)
  nnz = a.colptr[end]-1
  pnnz = nnz*plength
  resize!(a.rowval,nnz)
  δ = Int(length(a.nzval)/plength) - nnz
  if δ > 0
    for l in 1:plength
      Base._deleteat!(a.nzval,l*nnz+1,δ)
    end
  end
  data = reshape(a.nzval,nnz,plength)
  ConsecutiveParamSparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,data)
end

# csr: implentation needed
