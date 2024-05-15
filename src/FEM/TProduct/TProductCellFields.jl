struct TProductCellPoint{DS<:DomainStyle} <: CellDatum
  point::CellPoint{DS}
  single_points::Vector{<:CellPoint}
end

CellData.get_data(f::TProductCellPoint) = f.point
Base.length(a::TProductCellPoint) = length(a.single_points)

function CellData.get_triangulation(f::TProductCellPoint)
  s1 = first(f.single_points)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.single_points))
  trian
end

CellData.DomainStyle(::Type{TProductCellPoint{DS}}) where DS = DS()

# default behavior

function Arrays.evaluate!(cache,f::CellField,x::TProductCellPoint)
  evaluate!(cache,f,x.point)
end

struct TProductCellField{DS<:DomainStyle} <: CellField
  single_fields::Vector{<:CellField}
  domain_style::DS

  function TProductCellField(single_fields::Vector{<:CellField})
    @assert length(single_fields) > 0
    if any( map(i->DomainStyle(i)==ReferenceDomain(),single_fields) )
      domain_style = ReferenceDomain()
    else
      domain_style = PhysicalDomain()
    end
    new{typeof(domain_style)}(single_fields,domain_style)
  end
end

CellData.get_data(f::TProductCellField) = f.single_fields

function CellData.get_triangulation(f::TProductCellField)
  s1 = first(f.single_fields)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.single_fields))
  trian
end

CellData.DomainStyle(::Type{TProductCellField{DS}}) where DS = DS()
Base.length(a::TProductCellField) = length(a.single_fields)

function LinearAlgebra.dot(a::TProductCellField,b::TProductCellField)
  @check length(a) == length(b)
  return sum(map(dot,a.single_fields,b.single_fields))
end

# gradients

struct TProductGradientCellField <: CellField
  cell_data::CellField
  gradient_cell_data::CellField
end

CellData.get_data(a::TProductGradientCellField) = a.cell_data
CellData.DomainStyle(a::TProductGradientCellField) = DomainStyle(get_data(a))
CellData.get_triangulation(a::TProductGradientCellField) = get_triangulation(get_data(a))
get_gradient_data(a::TProductGradientCellField) = a.gradient_cell_data

(g::TProductGradientCellField)(x) = evaluate(g,x)

function Fields.gradient(f::TProductCellField)
  g = TProductCellField(gradient.(f.single_fields))
  return TProductGradientCellField(f,g)
end

function Fields.gradient(f::TProductFEBasis)
  dbasis = map(gradient,f.basis)
  trian = get_triangulation(f)
  g = TProductFEBasis(dbasis,trian)
  return TProductGradientCellField(f,g)
end

struct TProductGradientEval
  f::Vector
  g::Vector
  op
end

function TProductGradientEval(f::Vector,g::Vector)
  op = nothing
  TProductGradientEval(f,g,op)
end

CellData.get_data(a::TProductGradientEval) = a.f
get_gradient_data(a::TProductGradientEval) = a.g

# evaluations

const TProductCellDatum = Union{TProductFEBasis,TProductCellField}

function Arrays.return_cache(f::TProductCellDatum,x::TProductCellPoint)
  @assert length(f) == length(x)
  fitem = testitem(get_data(f))
  xitem = testitem(get_data(x))
  c1 = return_cache(fitem,xitem)
  fx1 = evaluate(fitem,xitem)
  cache = Vector{typeof(c1)}(undef,length(f))
  array = Vector{typeof(fx1)}(undef,length(f))
  return cache,array
end

function Arrays.evaluate!(_cache,f::TProductCellDatum,x::TProductCellPoint)
  cache,b = _cache
  @inbounds for i = 1:length(f)
    b[i] = evaluate!(cache[i],get_data(f)[i],get_data(x)[i])
  end
  return b
end

function Arrays.return_cache(k::Operation,f::TProductCellDatum...)
  D = length(first(f))
  @assert all(map(i -> length(get_data(i)) == D,f))
  fitem = map(testitem,get_data.(f))
  c1 = return_cache(k,fitem...)
  Fill(c1,D),Fill(k,D)
end

function Arrays.evaluate!(_cache,k::Operation,α::TProductCellDatum,β::TProductCellDatum)
  cache,K = _cache
  αβ = map(evaluate!,cache,K,get_data(α),get_data(β))
  TProductCellField(αβ)
end

function Arrays.return_cache(f::TProductGradientCellField,x::TProductCellPoint)
  cache = return_cache(get_data(f),x)
  gradient_cache = return_cache(get_gradient_data(f),x)
  return cache,gradient_cache
end

function Arrays.evaluate!(_cache,f::TProductGradientCellField,x::TProductCellPoint)
  cache,gradient_cache = _cache
  fx = evaluate!(cache,get_data(f),x)
  dfx = evaluate!(gradient_cache,get_gradient_data(f),x)
  return TProductGradientEval(fx,dfx)
end

function Arrays.return_cache(k::Operation,f::TProductGradientCellField...)
  cache = return_cache(k,map(get_data,f)...)
  gradient_cache = return_cache(k,map(get_gradient_data,f)...)
  return cache,gradient_cache
end

function Arrays.evaluate!(_cache,k::Operation,α::TProductGradientCellField,β::TProductGradientCellField)
  cache,gradient_cache = _cache
  αβ = evaluate!(cache,k,get_data(α),get_data(β))
  dαβ = evaluate!(gradient_cache,k,get_gradient_data(α),get_gradient_data(β))
  return TProductGradientCellField(αβ,dαβ)
end

# integration

function CellData.integrate(f::TProductCellDatum,a::TProductMeasure)
  map(integrate,get_data(f),a.measures_1d)
end

function CellData.integrate(f::TProductGradientCellField,a::TProductMeasure)
  fi = integrate(get_data(f),a)
  dfi = integrate(get_gradient_data(f),a)
  TProductGradientEval(fi,dfi)
end

# assembly

struct TProductSparseMatrixAssembler <: SparseMatrixAssembler
  assems_1d::Vector{GenericSparseMatrixAssembler}
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::TProductFESpace{D},
  test::TProductFESpace{D},
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()) where D

  assems_1d = map((U,V)->SparseMatrixAssembler(mat,vec,U,V,strategy),trial.spaces_1d,test.spaces_1d)
  TProductSparseMatrixAssembler(assems_1d)
end

function FESpaces.collect_cell_matrix(
  trial::TProductFESpace,
  test::TProductFESpace,
  a::Vector{<:DomainContribution})

  map(collect_cell_matrix,trial.spaces_1d,test.spaces_1d,a)
end

function FESpaces.collect_cell_vector(
  test::TProductFESpace,
  a::Vector{<:DomainContribution})

  map(collect_cell_vector,test.spaces_1d,a)
end

function FESpaces.collect_cell_matrix(
  trial::TProductFESpace,
  test::TProductFESpace,
  a::TProductGradientEval)

  f = collect_cell_matrix(trial,test,get_data(a))
  g = collect_cell_matrix(trial,test,get_gradient_data(a))
  TProductGradientEval(f,g,a.op)
end

function FESpaces.collect_cell_vector(
  test::TProductFESpace,
  a::TProductGradientEval)

  f = collect_cell_vector(test,get_data(a))
  g = collect_cell_vector(test,get_gradient_data(a))
  TProductGradientEval(f,g,a.op)
end

function FESpaces.allocate_vector(a::TProductSparseMatrixAssembler,vecdata::Vector)
  vecs_1d = map(allocate_vector,a.assems_1d,vecdata)
  vec = symbolic_kron(vecs_1d...)
  return TProductArray(vec,vecs_1d)
end

function FESpaces.assemble_vector!(b,a::TProductSparseMatrixAssembler,vecdata::Vector)
  map(b.arrays_1d,assemble_vector!,a.assems_1d,vecdata)
  numerical_kron!(b.array,b.arrays_1d...)
end

function FESpaces.assemble_vector_add!(b,a::TProductSparseMatrixAssembler,vecdata::Vector)
  map(b.arrays_1d,assemble_vector_add!,a.assems_1d,vecdata)
  numerical_kron!(b.array,b.arrays_1d...)
end

function FESpaces.assemble_vector(a::TProductSparseMatrixAssembler,vecdata::Vector)
  vecs_1d = map(assemble_vector,a.assems_1d,vecdata)
  vec = kron(vecs_1d...)
  return TProductArray(vec,vecs_1d)
end

function FESpaces.allocate_matrix(a::TProductSparseMatrixAssembler,matdata::Vector)
  mats_1d = map(allocate_matrix,a.assems_1d,matdata)
  mat = symbolic_kron(mats_1d...)
  return TProductArray(mat,mats_1d)
end

function FESpaces.assemble_matrix!(A,a::TProductSparseMatrixAssembler,matdata::Vector)
  map(assemble_matrix!,A.arrays_1d,a.assems_1d,matdata)
  numerical_kron!(A.array,A.arrays_1d...)
end

function FESpaces.assemble_matrix_add!(A,a::TProductSparseMatrixAssembler,matdata::Vector)
  map(assemble_matrix_add!,A.arrays_1d,a.assems_1d,matdata)
  numerical_kron!(A.array,A.arrays_1d...)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata::Vector)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata)
  mat = kron(mats_1d...)
  return TProductArray(mat,mats_1d)
end

struct TProductArray{T,N,A} <: AbstractArray{T,N}
  array::A
  arrays_1d::Vector{A}
  function TProductArray(array::A,arrays_1d::Vector{A}) where {T,N,A<:AbstractArray{T,N}}
    new{T,N,A}(array,arrays_1d)
  end
end

function TProductArray(arrays_1d::Vector{A}) where A
  array::A = kron(arrays_1d...)
  TProductArray(array,arrays_1d)
end

Base.size(a::TProductArray) = size(a.array)
Base.getindex(a::TProductArray,i...) = a.array[i...]
Base.iterate(a::TProductArray,i...) = iterate(a.array,i...)
Base.copy(a::TProductArray) = TProductArray(copy(a.array),a.arrays_1d)

Base.fill!(a::TProductArray,v) = fill!(a.array,v)

function LinearAlgebra.mul!(
  c::TProductArray,
  a::TProductArray,
  b::TProductArray,
  α::Number,β::Number)

  mul!(c.array,a.array,b.array,α,β)
end

function LinearAlgebra.axpy!(α::Number,a::TProductArray,b::TProductArray)
  axpy!(α,a.array,b.array)
end

for factorization in (:LU,:Cholesky)
  @eval begin
    function LinearAlgebra.ldiv!(m::$factorization,b::TProductArray)
      ldiv!(m,b.array)
      return b
    end
  end
end

function LinearAlgebra.ldiv!(a::TProductArray,m::Factorization,b::TProductArray)
  ldiv!(a.array,m,b.array)
  return a
end

function LinearAlgebra.rmul!(a::TProductArray,b::Number)
  rmul!(a.array,b)
  return a
end

function LinearAlgebra.lu(a::TProductArray)
  lu(a.array)
end

function LinearAlgebra.lu!(a::TProductArray,b::TProductArray)
  lu!(a.array,b.array)
end

const TProductSparseMatrix = TProductArray{T,2,A} where {T,A<:AbstractSparseMatrix}

SparseArrays.nnz(a::TProductSparseMatrix) = nnz(a.array)
SparseArrays.nzrange(a::TProductSparseMatrix,col::Int) = nzrange(a.array,col)
SparseArrays.rowvals(a::TProductSparseMatrix) = rowvals(a.array)
SparseArrays.nonzeros(a::TProductSparseMatrix) = a.array

function symbolic_kron(a::AbstractVector{T},b::AbstractVector{S}) where {T<:Number,S<:Number}
  c = Vector{promote_op(*,T,S)}(undef,length(a)*length(b))
  return c
end

@inline function numerical_kron!(c::Vector,a::AbstractVector{T},b::AbstractVector{S}) where {T<:Number,S<:Number}
  kron!(c,a,b)
end

function symbolic_kron(A::AbstractSparseMatrixCSC{T1,S1},B::AbstractSparseMatrixCSC{T2,S2}) where {T1,T2,S1,S2}
  mA,nA = size(A)
  mB,nB = size(B)
  mC,nC = mA*mB,nA*nB
  Tv = typeof(one(T1)*one(T2))
  Ti = promote_type(S1,S2)
  C = spzeros(Tv,Ti,mC,nC)
  sizehint!(C,nnz(A)*nnz(B))
  symbolic_kron!(C,A,B)
end

@inline function symbolic_kron!(C::SparseMatrixCSC,A::AbstractSparseMatrixCSC,B::AbstractSparseMatrixCSC)
  mA,nA = size(A)
  mB,nB = size(B)
  mC,nC = mA*mB,nA*nB

  msg = "target matrix needs to have size ($mC,$nC), but has size $(size(C))"
  @boundscheck size(C) == (mC,nC) || throw(DimensionMismatch(msg))

  rowvalC = rowvals(C)
  nzvalC = nonzeros(C)
  colptrC = getcolptr(C)

  nnzC = nnz(A)*nnz(B)
  resize!(nzvalC,nnzC)
  resize!(rowvalC,nnzC)

  col = 1
  @inbounds for j = 1:nA
    startA = getcolptr(A)[j]
    stopA = getcolptr(A)[j+1] - 1
    lA = stopA - startA + 1
    for i = 1:nB
      startB = getcolptr(B)[i]
      stopB = getcolptr(B)[i+1] - 1
      lB = stopB - startB + 1
      ptr_range = (1:lB) .+ (colptrC[col]-1)
      colptrC[col+1] = colptrC[col] + lA*lB
      col += 1
      for ptrA = startA : stopA
        ptrB = startB
        for ptr = ptr_range
          rowvalC[ptr] = (rowvals(A)[ptrA]-1)*mB + rowvals(B)[ptrB]
          ptrB += 1
        end
        ptr_range = ptr_range .+ lB
      end
    end
  end
  return C
end

@inline function numerical_kron!(C::SparseMatrixCSC,A::AbstractSparseMatrixCSC,B::AbstractSparseMatrixCSC)
  nA = size(A,2)
  nB = size(B,2)

  nzvalC = nonzeros(C)
  colptrC = getcolptr(C)

  col = 1
  @inbounds for j = 1:nA
    startA = getcolptr(A)[j]
    stopA = getcolptr(A)[j+1] - 1
    for i = 1:nB
      startB = getcolptr(B)[i]
      stopB = getcolptr(B)[i+1] - 1
      lB = stopB - startB + 1
      ptr_range = (1:lB) .+ (colptrC[col]-1)
      col += 1
      for ptrA = startA : stopA
        ptrB = startB
        for ptr = ptr_range
          nzvalC[ptr] = nonzeros(A)[ptrA] * nonzeros(B)[ptrB]
          ptrB += 1
        end
        ptr_range = ptr_range .+ lB
      end
    end
  end
  return C
end

function symbolic_kron(A::AbstractArray)
  A
end

function numerical_kron!(A::AbstractArray,B::AbstractArray)
  copyto!(A,B)
  A
end

function symbolic_kron(A::AbstractArray,B::AbstractArray,C::AbstractArray...)
  symbolic_kron(A,symbolic_kron(B,C...))
end

function numerical_kron!(A::AbstractArray,B::AbstractArray,C::AbstractArray...)
  numerical_kron!(A,numerical_kron!(B,C...))
end

# for gradients

function kronecker_gradients(f,g,op=nothing)
  Df = length(f)
  Dg = length(g)
  @check Df == Dg
  _kronecker_gradients(f,g,op,Val(Df))
end

_kronecker_gradients(f,g,::Val{1}) = g[1]
_kronecker_gradients(f,g,::Val{2}) = kron(g[1],f[2]) + kron(f[1],g[2])
_kronecker_gradients(f,g,::Val{3}) = kron(g[1],f[2],f[3]) + kron(f[1],g[2],f[3]) + kron(f[1],f[2],g[3])

_kronecker_gradients(f,g,::Nothing,::Val{d}) where d = _kronecker_gradients(f,g,Val(d))
_kronecker_gradients(f,g,op,::Val{d}) where d = op(kron(f...),_kronecker_gradients(f,g,Val(d)))

function symbolic_kron(f,g)
  Df = length(f)
  Dg = length(g)
  @check Df == Dg
  _symbolic_kron(f,g,Val(Df))
end

_symbolic_kron(f,g,::Val{1}) = symbolic_kron(g[1])
_symbolic_kron(f,g,::Val{2}) = symbolic_kron(g[1],f[2])
_symbolic_kron(f,g,::Val{3}) = symbolic_kron(g[1],f[2],f[3])

@inline function numerical_kron!(
  C::SparseMatrixCSC,
  vA::Vector{<:AbstractSparseMatrixCSC},
  vB::Vector{<:AbstractSparseMatrixCSC},
  op=nothing)

  _prod(f,g,::Val{1}) = g[1]
  _prod(f,g,::Val{2}) = g[1]*f[2] + f[1]*g[2]
  _prod(f,g,::Val{3}) = g[1]*f[2]*f[3] + f[1]*g[2]*f[3] + f[1]*f[2]*g[3]

  _prod(f,g,::Nothing,::Val{d}) where d = _prod(f,g,Val(d))
  _prod(f,g,op,::Val{d}) where d = op(prod(f),_prod(f,g,Val(d)))

  A = first(vA)
  B = first(vB)
  d = length(vA)

  nA = size(A,2)
  nB = size(B,2)

  nzvalvA = map(nonzeros,A)
  nzvalvB = map(nonzeros,B)
  cacheA = Vector{eltpye(nzvalC)}(undef,d)
  cacheB = Vector{eltpye(nzvalC)}(undef,d)

  nzvalC = nonzeros(C)
  colptrC = getcolptr(C)

  col = 1
  @inbounds for j = 1:nA
    startA = getcolptr(A)[j]
    stopA = getcolptr(A)[j+1] - 1
    for i = 1:nB
      startB = getcolptr(B)[i]
      stopB = getcolptr(B)[i+1] - 1
      lB = stopB - startB + 1
      ptr_range = (1:lB) .+ (colptrC[col]-1)
      col += 1
      for ptrA = startA : stopA
        ptrB = startB
        for ptr = ptr_range
          for di = 1:d
            cacheA[di] = nzvalvA[di][ptrA]
            cacheB[di] = nzvalvB[di][ptrB]
          end
          nzvalC[ptr] = _prod(cacheA,cacheB,op,Val(d))
          ptrB += 1
        end
        ptr_range = ptr_range .+ lB
      end
    end
  end
  return C
end

function FESpaces.allocate_vector(a::TProductSparseMatrixAssembler,vecdata::TProductGradientEval)
  vecs_1d = map(allocate_vector,a.assems_1d,vecdata.f)
  gradvecs_1d = map(allocate_vector,a.assems_1d,vecdata.g)
  vec = symbolic_kron(vecs_1d,gradvecs_1d)
  return TProductGradientArray(vec,vecs_1d,gradvecs_1d)
end

function FESpaces.assemble_vector!(b,a::TProductSparseMatrixAssembler,vecdata::TProductGradientEval)
  map(assemble_vector!,b.arrays_1d,a.assems_1d,vecdata.f)
  map(assemble_vector!,b.gradients_1d,a.assems_1d,vecdata.g)
  numerical_kron!(b.array,b.arrays_1d,b.gradients_1d,vecdata.op)
end

function FESpaces.assemble_vector_add!(b,a::TProductSparseMatrixAssembler,vecdata::TProductGradientEval)
  map(assemble_vector_add!,b.arrays_1d,a.assems_1d,vecdata.f)
  map(assemble_vector_add!,b.gradients_1d,a.assems_1d,vecdata.g)
  numerical_kron!(b.array,b.arrays_1d,b.gradients_1d,vecdata.op)
end

function FESpaces.assemble_vector(a::TProductSparseMatrixAssembler,vecdata::TProductGradientEval)
  vecs_1d = map(assemble_vector,a.assems_1d,vecdata.f)
  gradvecs_1d = map(assemble_vector,a.assems_1d,vecdata.g)
  vec = kronecker_gradients(vecs_1d,gradvecs_1d,vecdata.op)
  return TProductGradientArray(vec,vecs_1d,gradvecs_1d)
end

function FESpaces.allocate_matrix(a::TProductSparseMatrixAssembler,matdata::TProductGradientEval)
  mats_1d = map(allocate_matrix,a.assems_1d,matdata.f)
  gradmats_1d = map(allocate_matrix,a.assems_1d,matdata.g)
  mat = symbolic_kron(mats_1d,gradmats_1d)
  return TProductGradientArray(mat,mats_1d,gradmats_1d)
end

function FESpaces.assemble_matrix!(A,a::TProductSparseMatrixAssembler,matdata::TProductGradientEval)
  map(assemble_matrix!,A.arrays_1d,a.assems_1d,matdata.f)
  map(assemble_matrix!,A.gradients_1d,a.assems_1d,matdata.g)
  numerical_kron!(A.array,A.arrays_1d,A.gradients_1d,matdata.op)
end

function FESpaces.assemble_matrix_add!(A,a::TProductSparseMatrixAssembler,matdata::TProductGradientEval)
  map(assemble_matrix_add!,A.arrays_1d,a.assems_1d,matdata.f)
  map(assemble_matrix_add!,A.gradients_1d,a.assems_1d,matdata.g)
  numerical_kron!(A.array,A.arrays_1d,A.gradients_1d,matdata.op)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata::TProductGradientEval)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata.f)
  gradmats_1d = map(assemble_matrix,a.assems_1d,matdata.g)
  mat = kronecker_gradients(mats_1d,gradmats_1d,matdata.op)
  return TProductGradientArray(mat,mats_1d,gradmats_1d)
end

struct TProductGradientArray{T,N,A} <: AbstractArray{T,N}
  array::A
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
  function TProductGradientArray(
    array::A,arrays_1d::Vector{A},gradients_1d::Vector{A}) where {T,N,A<:AbstractArray{T,N}}
    new{T,N,A}(array,arrays_1d,gradients_1d)
  end
end

function TProductGradientArray(arrays_1d::Vector{A},gradients_1d::Vector{A}) where A
  array::A = kronecker_gradients(arrays_1d,gradients_1d)
  TProductGradientArray(array,arrays_1d)
end

Base.size(a::TProductGradientArray) = size(a.array)
Base.getindex(a::TProductGradientArray,i...) = a.array[i...]
Base.iterate(a::TProductGradientArray,i...) = iterate(a.array,i...)
Base.copy(a::TProductGradientArray) = TProductGradientArray(copy(a.array),a.arrays_1d,a.gradients_1d)

Base.fill!(a::TProductGradientArray,v) = fill!(a.array,v)

function LinearAlgebra.mul!(
  c::TProductGradientArray,
  a::TProductGradientArray,
  b::TProductGradientArray,
  α::Number,β::Number)

  mul!(c.array,a.array,b.array,α,β)
end

function LinearAlgebra.axpy!(α::Number,a::TProductGradientArray,b::TProductGradientArray)
  axpy!(α,a.array,b.array)
end

for factorization in (:LU,:Cholesky)
  @eval begin
    function LinearAlgebra.ldiv!(m::$factorization,b::TProductGradientArray)
      ldiv!(m,b.array)
      return b
    end
  end
end

function LinearAlgebra.ldiv!(a::TProductGradientArray,m::Factorization,b::TProductGradientArray)
  ldiv!(a.array,m,b.array)
  return a
end

function LinearAlgebra.rmul!(a::TProductGradientArray,b::Number)
  rmul!(a.array,b)
  return a
end

function LinearAlgebra.lu(a::TProductGradientArray)
  lu(a.array)
end

function LinearAlgebra.lu!(a::TProductGradientArray,b::TProductGradientArray)
  lu!(a.array,b.array)
end

const TProductGradientSparseMatrix = TProductGradientArray{T,2,A} where {T,A<:AbstractSparseMatrix}

SparseArrays.nnz(a::TProductGradientSparseMatrix) = nnz(a.array)
SparseArrays.nzrange(a::TProductGradientSparseMatrix,col::Int) = nzrange(a.array,col)
SparseArrays.rowvals(a::TProductGradientSparseMatrix) = rowvals(a.array)
SparseArrays.nonzeros(a::TProductGradientSparseMatrix) = a.array

# deal with cell field + gradient cell field
for op in (:+,:-)
  @eval ($op)(a::Vector,b::TProductGradientEval) = TProductGradientEval(b.f,b.g,$op)
  @eval ($op)(a::TProductGradientEval,b::Vector) = TProductGradientEval(a.f,a.g,$op)
end
