function Algebra.allocate_vector(::Type{V},n::Integer) where V<:AbstractParamContainer
  vector = zeros(eltype(V),n)
  allocate_param_array(vector,length(V))
end

function Algebra.allocate_in_range(matrix::ParamMatrix{T,A,L}) where {T,A,L}
  V = ParamVector{T,Vector{eltype(A)},L}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_domain(matrix::ParamMatrix{T,A,L}) where {T,A,L}
  V = ParamVector{T,Vector{eltype(A)},L}
  allocate_in_domain(V,matrix)
end

function Algebra.allocate_vector(
  ::Type{<:ParamBlockVector{T,V}},
  indices::BlockedUnitRange) where {T,V}

  mortar(map(ids -> allocate_vector(V,ids),blocks(indices)))
end

function Algebra.allocate_in_range(matrix::ParamBlockMatrix{T,A,L}) where {T,A,L}
  BV = BlockVector{T,Vector{ParamVector{T,Vector{eltype(A)},L}}}
  V = ParamBlockVector{T,Vector{eltype(A)},L,BV}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_domain(matrix::ParamBlockMatrix{T,A,L}) where {T,A,L}
  BV = BlockVector{T,Vector{ParamVector{T,Vector{eltype(A)},L}}}
  V = ParamBlockVector{T,Vector{eltype(A)},L,BV}
  allocate_in_domain(V,matrix)
end

function Algebra.nz_allocation(a::Algebra.ArrayCounter{<:ParamVector{T,A,L}}) where {T,A,L}
  elA = eltype(A)
  v = fill!(similar(elA,map(length,a.axes)),zero(T))
  allocate_param_array(v,L)
end

@inline function Algebra.add_entry!(combine::Function,A::ParamVector,v::Number,i)
  for k = eachindex(A)
    ai = A[k][i]
    A[k][i] = combine(ai,v)
  end
  A
end

function Algebra.is_entry_stored(::Type{<:ParamMatrix{T,A,L}},i,j) where {T,A,L}
  is_entry_stored(eltype(A),i,j)
end

@inline function Algebra.add_entry!(combine::Function,A::ParamMatrix,v::Number,i,j)
  for k = eachindex(A)
    aij = A[k][i,j]
    A[k][i,j] = combine(aij,v)
  end
  A
end

@inline function Algebra.add_entry!(::typeof(+),a::Algebra.AllocationCOO{T},::Nothing,i,j) where T<:ParamMatrix
  if Algebra.is_entry_stored(T,i,j)
    a.counter.nnz = a.counter.nnz + 1
    k = a.counter.nnz
    a.I[k] = i
    a.J[k] = j
  end
  nothing
end

@inline function Algebra.add_entry!(::typeof(+),a::Algebra.AllocationCOO{T},v,i,j) where T<:ParamMatrix
  if Algebra.is_entry_stored(T,i,j)
    a.counter.nnz = a.counter.nnz + 1
    k = a.counter.nnz
    a.I[k] = i
    a.J[k] = j
    for l = 1:length(T)
      a.V[l][k] = v[l]
    end
  end
  nothing
end

function Algebra.copy_entries!(
  a::ParamMatrix{Ta,Vector{<:AbstractSparseMatrix}} where Ta,
  b::ParamMatrix{Tb,Vector{<:AbstractSparseMatrix}} where Tb)
  na = nonzeros(a)
  nb = nonzeros(b)
  if na !== nb
    copyto!(na,nb)
  end
end

function Algebra.allocate_coo_vectors(
  ::Type{<:ParamMatrix{Tv,<:Vector{<:AbstractSparseMatrix{Tv,Ti}},L}},
  n::Integer) where {Tv,Ti,L}
  I = zeros(Ti,n)
  J = zeros(Ti,n)
  V = zeros(Tv,n)
  PV = allocate_param_array(V,L)
  I,J,PV
end

function Algebra.sparse_from_coo(::Type{ParamMatrix{T,A,L}},I,J,V::ParamArray,m,n) where {T,A,L}
  elA = eltype(A)
  psparse = map(1:L) do k
    Algebra.sparse_from_coo(elA,I,J,V[k],m,n)
  end
  ParamArray(psparse)
end

function Algebra.finalize_coo!(::Type{ParamMatrix{T,A,L}},I,J,V::ParamArray,m,n) where {T,A,L}
  elA = eltype(A)
  map(1:L) do k
    Algebra.finalize_coo!(elA,I,J,V[k],m,n)
  end
end

function Algebra.nz_index(a::ParamMatrix,i0,i1)
  nz_index(first(a),i0,i1)
end

function Algebra.push_coo!(::Type{ParamMatrix{T,A,L}},I,J,V::ParamArray,i,j,v) where {T,A,L}
  elA = eltype(A)
  map(1:L) do k
    Algebra.push_coo!(elA,I,J,V[k],i,j,v)
  end
end

struct ParamCounter{C,L}
  counter::C
  function ParamCounter(counter::C,::Val{L}) where {C,L}
    new{C,L}(counter)
  end
end

function ParamCounter(counter::C,L::Integer) where C
  ParamCounter(counter,Val(L))
end

Algebra.LoopStyle(::Type{ParamCounter{C,L}}) where {C,L} = LoopStyle(C)

@inline function Algebra.add_entry!(::typeof(+),a::ParamCounter,v,i,j)
  add_entry!(+,a.counter,v,i,j)
end

function Algebra.nz_counter(
  builder::SparseMatrixBuilder{<:ParamMatrix{T,A,L}},
  axes) where {T,A,L}

  elA = eltype(A)
  elb = SparseMatrixBuilder(elA)
  counter = nz_counter(elb,axes)
  ParamCounter(counter,L)
end

function Algebra.nz_allocation(a::ParamCounter{C,L}) where {C,L}
  inserter = nz_allocation(a.counter)
  ParamInserter(inserter,L)
end

# CSC

function ParamInserter(inserter::Algebra.InserterCSC,L::Integer)
  ParamInserterCSC(inserter,Val(L))
end

struct ParamInserterCSC{Tv,Ti,P}
  nrows::Int
  ncols::Int
  colptr::Vector{Ti}
  colnnz::Vector{Ti}
  rowval::Vector{Ti}
  nzval::P
  function ParamInserterCSC(inserter::Algebra.InserterCSC{Tv,Ti},::Val{L}) where {Tv,Ti,L}
    @unpack nrows,ncols,colptr,colnnz,rowval,nzval = inserter
    pnzval = allocate_param_array(nzval,L)
    P = typeof(pnzval)
    new{Tv,Ti,P}(nrows,ncols,colptr,colnnz,rowval,pnzval)
  end
end

Algebra.LoopStyle(::Type{<:ParamInserterCSC}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC{Tv,Ti},v::Nothing,i,j)  where {Tv,Ti}
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

@noinline function Algebra.add_entry!(
  ::typeof(+),a::ParamInserterCSC{Tv,Ti,P},v::AbstractParamContainer{Tv},i,j) where {Tv,Ti,P}
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for l = 1:length(P)
      a.nzval[l][p] = v[l]
    end
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check  pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
      @inbounds for l = 1:length(P)
        a.nzval[l][o] = a.nzval[l][k]
      end
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    @inbounds for l = 1:length(P)
      a.nzval[l][p] = v[l]
    end
  else
    # update existing entry
    @inbounds for l = 1:length(P)
      a.nzval[l][p] += v[l]
    end
  end
  nothing
end

function Algebra.create_from_nz(a::ParamInserterCSC{Tv,Ti,P}) where {Tv,Ti,P}
  k = 1
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      @inbounds for l = 1:length(P)
        a.nzval[l][k] = a.nzval[l][p]
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
  resize!(a.rowval,nnz)
  resize!(a.nzval,nnz)
  csc = map(1:length(P)) do l
    SparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,a.nzval[l])
  end
  ParamArray(csc)
end

# CSSR

function ParamInserter(inserter::Algebra.CSRR,L::Integer)
  ParamCSSR(inserter,Val(L))
end

struct ParamCSSR{Tv,Ti,P}
  nrows::Int
  ncols::Int
  colptr::Vector{Ti}
  colnnz::Vector{Ti}
  rowval::Vector{Ti}
  nzval::ParamVector{Tv}
  work::Vector{Ti}
  function ParamCSSR(inserter::Algebra.CSRR{Tv,Ti},::Val{L}) where {Tv,Ti,L}
    @unpack nrows,ncols,colptr,colnnz,rowval,nzval,work = inserter
    pnzval = allocate_param_array(nzval,L)
    P = typeof(pnzval)
    new{Tv,Ti,P}(nrows,ncols,colptr,colnnz,rowval,pnzval,work)
  end
end

Algebra.LoopStyle(::Type{<:ParamCSSR}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::ParamCSSR{Tv,Ti},v::Nothing,i,j) where {Tv,Ti}
  p = a.rowptrs[i]
  a.colvals[p] = j
  a.rowptrs[i] = p+Ti(1)
  nothing
end

@inline function Algebra.add_entry!(
  ::typeof(+),a::ParamCSSR{Tv,Ti,P},v::AbstractParamContainer{Tv},i,j) where {Tv,Ti,P}
  p = a.rowptrs[i]
  a.colvals[p] = j
  @inbounds for l = 1:length(P)
    a.nzvals[l][p] = v[l]
  end
  a.rowptrs[i] = p+Ti(1)
  nothing
end

function Algebra.create_from_nz(a::ParamCSSR{Tv,Ti,P}) where {Tv,Ti,P}
  rewind_ptrs!(a.rowptrs)
  colptrs = Vector{Ti}(undef,a.ncols+1)
  cscnnz = Algebra._csrr_to_csc_count!(colptrs,a.rowptrs,a.colvals,a.nzvals,a.work)
  rowvals = Vector{Ti}(undef,cscnnz)
  nzvalscsc = Vector{Tv}(undef,cscnnz)
  pnzvalscsc = allocate_param_array(nzvalscsc,length(P))
  Algebra._csrr_to_csc_fill!(colptrs,rowvals,pnzvalscsc,a.rowptrs,a.colvals,a.nzvals)
  SparseMatrixCSC(a.nrows,a.ncols,colptrs,rowvals,pnzvalscsc)
end

function Algebra._csrr_to_csc_count!(
  colptrs::Vector{Ti},
  rowptrs::Vector{Tj},
  colvals::Vector{Tj},
  nzvalscsr::ParamVector{Tv},
  work::Vector{Tj}) where {Ti,Tj,Tv}

  nrows = length(rowptrs)-1
  ncols = length(colptrs)-1
  if nrows == 0 || ncols == 0
    fill!(colptrs, Ti(1))
    return Tj(0)
  end

  # Convert csrr to csru by identifying repeated cols with array work.
  # At the same time, count number of unique rows in colptrs shifted by one.
  fill!(colptrs, Ti(0))
  fill!(work, Tj(0))
  writek = Tj(1)
  newcsrrowptri = Ti(1)
  origcsrrowptri = Tj(1)
  origcsrrowptrip1 = rowptrs[2]
  @inbounds for i in 1:nrows
    for readk in origcsrrowptri:(origcsrrowptrip1-Tj(1))
      j = colvals[readk]
      if work[j] < newcsrrowptri
        work[j] = writek
        if writek != readk
          colvals[writek] = j
          for l = eachindex(nzvalscsr)
            nzvalscsr[l][writek] = nzvalscsr[l][readk]
          end
        end
        writek += Tj(1)
        colptrs[j+1] += Ti(1)
      else
        klt = work[j]
        for l = eachindex(nzvalscsr)
          nzvalscsr[l][klt] = +(nzvalscsr[l][klt], nzvalscsr[l][readk])
        end
      end
    end
    newcsrrowptri = writek
    origcsrrowptri = origcsrrowptrip1
    origcsrrowptrip1 != writek && (rowptrs[i+1] = writek)
    i < nrows && (origcsrrowptrip1 = rowptrs[i+2])
  end

  # Convert colptrs from counts to ptrs shifted by one
  # (ptrs will be corrected below)
  countsum = Tj(1)
  colptrs[1] = Ti(1)
  @inbounds for j in 2:(ncols+1)
    overwritten = colptrs[j]
    colptrs[j] = countsum
    countsum += overwritten
    @check Base.hastypemax(Ti) && (countsum <= typemax(Ti))
  end

  cscnnz = countsum - Tj(1)
  cscnnz
end

function Algebra._csrr_to_csc_fill!(
  colptrs::Vector{Ti},rowvals::Vector{Ti},nzvalscsc::ParamVector{Tv},
  rowptrs::Vector{Tj},colvals::Vector{Tj},nzvalscsr::ParamVector{Tv}) where {Ti,Tj,Tv}

  nrows = length(rowptrs)-1
  ncols = length(colptrs)-1
  if nrows == 0 || ncols == 0
    return nothing
  end

  # From csru to csc
  # Tracking write positions in colptrs corrects
  # the column pointers to the final value.
  @inbounds for i in 1:nrows
    for csrk in rowptrs[i]:(rowptrs[i+1]-Tj(1))
      j = colvals[csrk]
      csck = colptrs[j+1]
      colptrs[j+1] = csck + Ti(1)
      rowvals[csck] = i
      for l = eachindex(nzvalscsc)
        nzvalscsc[l][csck] = nzvalscsr[l][csrk]
      end
    end
  end

  nothing
end

# CSR

function Algebra.create_from_nz(a::Algebra.NzAllocationCSR{Bi,<:ParamInserterCSC}) where Bi
  Atcsc = create_from_nz(a.csc)
  Acsc = transpose(Atcsc)
  Acsr = map(1:length(Acsc)) do l
    SparseMatrixCSR{Bi}(Acsc[l])
  end
  ParamArray(Acsr)
end
