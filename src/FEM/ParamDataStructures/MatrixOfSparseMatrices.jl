"""
    struct MatrixOfSparseMatricesCSC{Tv,Ti<:Integer,L} <: ParamSparseMatrixCSC{Tv,Ti,L} end

Represents a vector of sparse matrices in CSC format. For sake of coherence, an
instance of `MatrixOfSparseMatricesCSC` inherits from AbstractMatrix{<:SparseMatrixCSC{Tv,Ti}
rather than AbstractVector{<:SparseMatrixCSC{Tv,Ti}, but should conceptually be
thought as an AbstractVector{<:SparseMatrixCSC{Tv,Ti}.

"""
struct MatrixOfSparseMatricesCSC{Tv,Ti<:Integer,L} <: ParamSparseMatrixCSC{Tv,Ti,L}
  m::Int64
  n::Int64
  colptr::Vector{Ti}
  rowval::Vector{Ti}
  data::Matrix{Tv}
  function MatrixOfSparseMatricesCSC(
    m::Int64,
    n::Int64,
    colptr::Vector{Ti},
    rowval::Vector{Ti},
    data::Matrix{Tv}
    ) where {Tv,Ti}

    L = size(data,2)
    new{Tv,Ti,L}(m,n,colptr,rowval,data)
  end
end

SparseArrays.getcolptr(A::MatrixOfSparseMatricesCSC) = A.colptr
SparseArrays.rowvals(A::MatrixOfSparseMatricesCSC) = A.rowval
SparseArrays.nonzeros(A::MatrixOfSparseMatricesCSC) = A.data
SparseArrays.nnz(A::MatrixOfSparseMatricesCSC) = Int(getcolptr(A)[innersize(A,2)+1])-1
SparseArrays.nonzeros(A::MatrixOfSparseMatricesCSC) = ConsecutiveArrayOfArrays(A.data)
_nonzeros(A::MatrixOfSparseMatricesCSC,iblock::Integer...) = @inbounds getindex(A.data,:,iblock...)

function SparseArrays.findnz(A::MatrixOfSparseMatricesCSC{Tv,Ti,L}) where {Tv,Ti,L}
  numnz = nnz(A)
  nz = nonzeros(A)
  I = Vector{Ti}(undef,numnz)
  J = Vector{Ti}(undef,numnz)
  V = Vector{Tv}(undef,numnz)
  pV = array_of_consecutive_arrays(V,L)

  count = 1
  @inbounds for col = 1 : innersize(A,2),k = getcolptr(A)[col] : (getcolptr(A)[col+1]-1)
    I[count] = rowvals(S)[k]
    J[count] = col
    @views pV.data[count,:] = nz.data[k,:]
    count += 1
  end

  return (I,J,pV)
end

function MatrixOfSparseMatricesCSC(A::AbstractVector{SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
  m,n = innersize(A)
  colptr,rowval,data = innerpattern(A)
  B = ArrayOfSimilarArrays(data)
  MatrixOfSparseMatricesCSC(m,n,colptr,rowval,B.data)
end

function innerpattern(A::AbstractVector{<:SparseMatrixCSC})
  @check !isempty(A)
  colptr,rowval = first(A).colptr,first(A).rowval
  if (any(A->A.colptr != colptr, A) || any(A->A.rowval != rowval, A))
    throw(DimensionMismatch("Sparsity pattern of element csc matrices is not equal"))
  end
  data = map(nonzeros,A)
  return colptr,rowval,data
end

Base.size(A::MatrixOfSparseMatricesCSC) = (param_length(A),param_length(A))

@inline function ArraysOfArrays.innersize(A::MatrixOfSparseMatricesCSC)
  (A.m,A.n)
end

@inline function inneraxes(A::MatrixOfSparseMatricesCSC)
  Base.OneTo.(innersize(A))
end

param_entry(A::MatrixOfSparseMatricesCSC,i::Integer...) = A.data[i...,:]

Base.@propagate_inbounds function Base.getindex(A::MatrixOfSparseMatricesCSC,i::Vararg{Integer,2})
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    param_getindex(A,iblock)
  else
    spzeros(innersize(A))
  end
end

Base.@propagate_inbounds function param_getindex(A::MatrixOfSparseMatricesCSC,i::Integer)
  SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,_nonzeros(A,i))
end

Base.@propagate_inbounds function Base.setindex!(A::MatrixOfSparseMatricesCSC,v,i::Vararg{Integer,2})
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  irow==icol && param_setindex!(A,v,i)
end

Base.@propagate_inbounds function param_setindex!(A::MatrixOfSparseMatricesCSC,v,i::Integer)
  @boundscheck iblock ≤ param_length(A)
  setindex!(_nonzeros(A,iblock),v)
end

function Base.similar(
  A::MatrixOfSparseMatricesCSC{Tv,Ti},
  ::Type{<:SparseMatrixCSC{Tv′,Ti′}}
  ) where {Tv,Ti,Tv′,Ti′}

  colptr = similar(A.colptr,Ti′)
  rowval = similar(A.rowval,Ti′)
  nzvals = similar(A.data,Tv′)
  MatrixOfSparseMatricesCSC(A.m,A.n,colptr,rowval,nzvals)
end

function Base.copyto!(A::MatrixOfSparseMatricesCSC,B::MatrixOfSparseMatricesCSC)
  @check size(A) == size(B)
  copyto!(A.colptr,B.colptr)
  copyto!(A.rowval,B.rowval)
  copyto!(A.data,B.data)
  A
end

# small hack
Base.iszero(A::MatrixOfSparseMatricesCSC) = (nnz(A) == 0)

function LinearAlgebra.diag(A::MatrixOfSparseMatricesCSC{Tv,Ti},d::Integer=0) where {Tv,Ti}
  m,n = innersize(A)
  k = Int(d)
  l = k < 0 ? min(m+k,n) : min(n-k,m)
  r,c = k <= 0 ? (-k,0) : (0,k) # start row/col -1
  ind = Vector{Ti}()
  val = zeros(Tv,l,param_length(A))
  for i in 1:l
    r += 1; c += 1
    r1 = Int(getcolptr(A)[c])
    r2 = Int(getcolptr(A)[c+1]-1)
    r1 > r2 && continue
    r1 = searchsortedfirst(rowvals(A),r,r1,r2,SparseArrays.Forward)
    ((r1 > r2) || (rowvals(A)[r1] != r)) && continue
    push!(ind,i)
    val[length(ind),:] = A.data[r1,:]
  end
  return VectorOfSparseVectors(l,ind,val)
end

# some sparse operations

function array_of_zero_arrays(a::SparseMatrixCSC,plength::Integer)
  A = array_of_similar_arrays(a,plength)
  LinearAlgebra.fillstored!(A,zero(eltype(a)))
  return A
end

function LinearAlgebra.fillstored!(A::MatrixOfSparseMatricesCSC,z::Number)
  fill!(A.data,z)
  return A
end

# small hack, we shouldn't be able to fill an abstract array with a non-scalar
function LinearAlgebra.fillstored!(A::MatrixOfSparseMatricesCSC,z::AbstractMatrix{<:Number})
  @check all(z.==first(z))
  LinearAlgebra.fillstored!(A,first(z))
  return A
end

function recast(a::AbstractMatrix,A::SparseMatrixCSC)
  @check size(a,1) == nnz(A)
  B = map(v -> recast(v,A),collect.(eachcol(a)))
  return MatrixOfSparseMatricesCSC(B)
end

function recast(a::AbstractArray,A::ParamSparseMatrixCSC)
  recast(a,param_getindex(A,1))
end

fast_issymmetric(A::SparseArrays.AbstractSparseMatrixCSC) = fast_is_hermsym(A, identity)

fast_ishermitian(A::SparseArrays.AbstractSparseMatrixCSC) = fast_is_hermsym(A, conj)

function fast_is_hermsym(A::SparseArrays.AbstractSparseMatrixCSC,check::Function)
  m, n = size(A)
  if m != n; return false; end

  colptr = getcolptr(A)
  rowval = rowvals(A)
  nzval = nonzeros(A)
  tracker = copy(getcolptr(A))
  for col = axes(A, 2)
    # `tracker` is updated such that, for symmetric matrices,
    # the loop below starts from an element at or below the
    # diagonal element of column `col`"
    for p = tracker[col]:colptr[col+1]-1
      val = nzval[p]
      row = rowval[p]

      # Ignore stored zeros
      if iszero(val)
          continue
      end

      # If the matrix was symmetric we should have updated
      # the tracker to start at the diagonal or below. Here
      # we are above the diagonal so the matrix can't be symmetric.
      if row < col
        return false
      end

      # Diagonal element
      if row == col
        if val != check(val)
          return false
        end
      else
        offset = tracker[row]

        # If the matrix is unsymmetric, there might not exist
        # a rowval[offset]
        if offset > length(rowval)
          return false
        end

        row2 = rowval[offset]

        # row2 can be less than col if the tracker didn't
        # get updated due to stored zeros in previous elements.
        # We therefore "catch up" here while making sure that
        # the elements are actually zero.
        while row2 < col
          if SparseArrays._isnotzero(nzval[offset])
            return false
          end
          offset += 1
          row2 = rowval[offset]
          tracker[row] += 1
        end

        # Non zero A[i,j] exists but A[j,i] does not exist
        if row2 > col
          return false
        end

        # A[i,j] and A[j,i] exists
        if row2 == col
          if !isapprox(val,check(nzval[offset]))
            return false
          end
          tracker[row] += 1
        end
      end
    end
  end
  return true
end

function fast_cholesky(A::AbstractSparseMatrix;kwargs...)
  @abstractmethod
end

function fast_cholesky(A::SparseArrays.AbstractSparseMatrixCSC;kwargs...)
  if !ishermitian(A) && fast_ishermitian(A)
    @views @inbounds A .= (A + A') / 2
    @check ishermitian(A)
  end
  cholesky(A;kwargs...)
end

function LinearAlgebra.cholesky(A::MatrixOfSparseMatricesCSC;kwargs...)
  ParamContainer(fast_cholesky.(param_data(A);kwargs...))
end
