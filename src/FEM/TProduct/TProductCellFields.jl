struct TProductCellPoint{DS<:DomainStyle} <: CellDatum
  single_points::Vector{<:CellPoint}
  domain_style::DS

  function TProductCellPoint(single_points::Vector{<:CellPoint})
    @assert length(single_points) > 0
    if any( map(i->DomainStyle(i)==ReferenceDomain(),single_points) )
      domain_style = ReferenceDomain()
    else
      domain_style = PhysicalDomain()
    end
    new{typeof(domain_style)}(single_points,domain_style)
  end
end

CellData.get_data(f::TProductCellPoint) = f.single_points
MultiField.num_fields(a::TProductCellPoint) = length(a.single_points)
Base.getindex(a::TProductCellPoint,i::Integer) = a.single_points[i]
Base.iterate(a::TProductCellPoint)  = iterate(a.single_points)
Base.iterate(a::TProductCellPoint,state) = iterate(a.single_points,state)
Base.length(a::TProductCellPoint) = num_fields(a)

function CellData.get_triangulation(f::TProductCellPoint)
  s1 = first(f.single_points)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.single_points))
  trian
end

CellData.DomainStyle(::Type{TProductCellPoint{DS}}) where DS = DS()

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
MultiField.num_fields(a::TProductCellField) = length(a.single_fields)
Base.getindex(a::TProductCellField,i::Integer) = a.single_fields[i]
Base.iterate(a::TProductCellField)  = iterate(a.single_fields)
Base.iterate(a::TProductCellField,state) = iterate(a.single_fields,state)
Base.length(a::TProductCellField) = num_fields(a)

function LinearAlgebra.dot(a::TProductCellField,b::TProductCellField)
  @check num_fields(a) == num_fields(b)
  return sum(map(dot,a.single_fields,b.single_fields))
end

struct TProductFESpace{D} <: SingleFieldFESpace
  space::SingleFieldFESpace
  spaces_1d::Vector{<:SingleFieldFESpace}
  dof_permutation::Array{Int,D}
end

function TProductFESpace(
  model::TProductModel,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args
  cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
  cell_reffes_1d = map(model->ReferenceFE(model,basis,T,order;reffe_kwargs...),model.models_1d)
  space = FESpace(model,cell_reffe;kwargs...)
  spaces_1d = map(FESpace,model.models_1d,cell_reffes_1d) # is it ok to eliminate the kwargs?
  perm = get_tp_dof_permutation(T,model.models_1d,spaces_1d,order)
  TProductFESpace(space,spaces_1d,perm)
end

FESpaces.get_triangulation(f::TProductFESpace) = get_triangulation(f.space)

FESpaces.get_free_dof_ids(f::TProductFESpace) = get_free_dof_ids(f.space)

function FESpaces.get_vector_type(f::TProductFESpace{D}) where D
  T = eltype(get_vector_type(f.space))
  Array{T,D}
end

FESpaces.get_dof_value_type(f::TProductFESpace) = get_dof_value_type(f.space)

FESpaces.ConstraintStyle(f::TProductFESpace) = ConstraintStyle(f.space)

# struct TProductFEBasisComponent{B} <: FEBasis
#   cell_basis::AbstractArray
#   single_field::B
#   fieldid::Int
#   nfields::Int
#   function TProductFEBasisComponent(
#     single_field::FESpaces.SingleFieldFEBasis,fieldid::Integer,nfields::Integer)
#     function block_dofs(cell_bs,::FESpaces.TestBasis,fieldid,nfields)
#       cell_basis = lazy_map(BlockMap(nfields,fieldid),cell_bs)
#     end
#     function block_dofs(cell_bs,::FESpaces.TrialBasis,fieldid,nfields)
#       cell_basis = lazy_map(BlockMap((1,nfields),fieldid),cell_bs)
#     end
#     B = typeof(single_field)
#     cell_bs = get_data(single_field)
#     bsty = BasisStyle(single_field)
#     cell_basis = block_dofs(cell_bs,bsty,fieldid,nfields)
#     new{B}(cell_basis,single_field,fieldid,nfields)
#   end
# end

# CellData.get_data(f::TProductFEBasisComponent) = f.cell_basis
# CellData.get_triangulation(f::TProductFEBasisComponent) = get_triangulation(f.single_field)
# FESpaces.BasisStyle(::Type{<:TProductFEBasisComponent{B}}) where B = BasisStyle(B)
# CellData.DomainStyle(::Type{<:TProductFEBasisComponent{B}}) where B = DomainStyle(B)
# function FESpaces.CellData.similar_cell_field(
#   f::TProductFEBasisComponent,cell_data,trian,ds::DomainStyle)
#   @notimplemented
# end
# function FESpaces.similar_fe_basis(
#   f::TProductFEBasisComponent,cell_data,trian,bs::BasisStyle,ds::DomainStyle)
#   @notimplemented
# end

# function CellData.change_domain(
#   a::TProductFEBasisComponent,
#   tdomain::DomainStyle)
#   sf = change_domain(a.single_field,tdomain)
#   TProductFEBasisComponent(sf,a.fieldid,a.nfields)
# end

# function CellData.change_domain(
#   a::TProductFEBasisComponent,
#   ttrian::Triangulation,
#   tdomain::DomainStyle)
#   sf = change_domain(a.single_field,ttrian,tdomain)
#   TProductFEBasisComponent(sf,a.fieldid,a.nfields)
# end

# function FESpaces.get_fe_basis(f::TProductFESpace)
#   nfields = length(f.spaces_1d)
#   all_febases = TProductFEBasisComponent[]
#   for field_i in 1:nfields
#     dv_i = get_fe_basis(f.spaces_1d[field_i])
#     @assert FESpaces.BasisStyle(dv_i) == FESpaces.TestBasis()
#     dv_i_b = TProductFEBasisComponent(dv_i,field_i,nfields)
#     push!(all_febases,dv_i_b)
#   end
#   TProductCellField(all_febases)
# end

# function FESpaces.get_trial_fe_basis(f::TProductFESpace)
#   nfields = length(f.spaces_1d)
#   all_febases = TProductFEBasisComponent[]
#   for field_i in 1:nfields
#     du_i = get_trial_fe_basis(f.spaces_1d[field_i])
#     @assert FESpaces.BasisStyle(du_i) == FESpaces.TrialBasis()
#     du_i_b = TProductFEBasisComponent(du_i,field_i,nfields)
#     push!(all_febases,du_i_b)
#   end
#   TProductCellField(all_febases)
# end
struct TProductFEBasis{DS,BS} <: FEBasis
  basis::ArrayBlock
  trian::Triangulation
  domain_style::DS
  basis_style::BS
end

function TProductFEBasis(basis::ArrayBlock,trian::TProductTriangulation)
  b1 = testitem(basis)
  DS = DomainStyle(b1)
  BS = BasisStyle(b1)
  @check all(map(i -> DS===DomainStyle(i) && BS===BasisStyle(i),basis.array))
  TProductFEBasis(basis,trian,DS,BS)
end

CellData.get_data(f::TProductFEBasis) = f.basis.array
CellData.get_triangulation(f::TProductFEBasis) = f.trian
FESpaces.BasisStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = BS
CellData.DomainStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = DS
MultiField.num_fields(a::TProductFEBasis) = length(get_data(a))
Base.length(a::TProductFEBasis) = num_fields(a)

function FESpaces.get_fe_basis(f::TProductFESpace)
  nfields = length(f.spaces_1d)
  touched = fill(true,nfields)
  basis = map(get_fe_basis,f.spaces_1d)
  bbasis = ArrayBlock(basis,touched)
  trian = get_triangulation(f)
  TProductFEBasis(bbasis,trian)
end

function FESpaces.get_trial_fe_basis(f::TProductFESpace)
  nfields = length(f.spaces_1d)
  touched = fill(true,nfields)
  basis = map(get_trial_fe_basis,f.spaces_1d)
  bbasis = ArrayBlock(basis,touched)
  trian = get_triangulation(f)
  TProductFEBasis(bbasis,trian)
end

# gradients

struct TProductGradient <: CellField
  cell_data::CellField
  gradient_cell_data::CellField
end

CellData.get_data(a::TProductGradient) = a.cell_data
CellData.DomainStyle(a::TProductGradient) = DomainStyle(get_data(a))
CellData.get_triangulation(a::TProductGradient) = get_triangulation(get_data(a))
get_gradient_data(a::TProductGradient) = a.gradient_cell_data

(g::TProductGradient)(x) = evaluate(g,x)

function Fields.gradient(f::TProductCellField)
  g = TProductCellField(gradient.(f.single_fields))
  return TProductGradient(f,g)
end

function Fields.gradient(f::TProductFEBasis)
  dbasis = ArrayBlock(map(gradient,f.basis.array),f.basis.touched)
  trian = get_triangulation(f)
  g = TProductFEBasis(dbasis,trian)
  return TProductGradient(f,g)
end

struct TProductGradientArray
  f::VectorBlock
  g::VectorBlock
end

CellData.get_data(a::TProductGradientArray) = a.f
get_gradient_data(a::TProductGradientArray) = a.g

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
  touched = fill(true,length(f))
  b = ArrayBlock(array,touched)
  return cache,b
end

function Arrays.evaluate!(_cache,f::TProductCellDatum,x::TProductCellPoint)
  cache,b = _cache
  @inbounds for i = 1:length(f)
    b.array[i] = evaluate!(cache[i],get_data(f)[i],get_data(x)[i])
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

function Arrays.return_cache(f::TProductGradient,x::TProductCellPoint)
  cache = return_cache(get_data(f),x)
  gradient_cache = return_cache(get_gradient_data(f),x)
  return cache,gradient_cache
end

function Arrays.evaluate!(_cache,f::TProductGradient,x::TProductCellPoint)
  cache,gradient_cache = _cache
  fx = evaluate!(cache,get_data(f),x)
  dfx = evaluate!(gradient_cache,get_gradient_data(f),x)
  return TProductGradientArray(fx,dfx)
end

function Arrays.return_cache(k::Operation,f::TProductGradient...)
  cache = return_cache(k,map(get_data,f)...)
  gradient_cache = return_cache(k,map(get_gradient_data,f)...)
  return cache,gradient_cache
end

function Arrays.evaluate!(_cache,k::Operation,α::TProductGradient,β::TProductGradient)
  cache,gradient_cache = _cache
  αβ = evaluate!(cache,k,get_data(α),get_data(β))
  dαβ = evaluate!(gradient_cache,k,get_gradient_data(α),get_gradient_data(β))
  return TProductGradient(αβ,dαβ)
end

# integration

function CellData.integrate(f::TProductCellDatum,a::TProductMeasure)
  array = map(integrate,get_data(f),a.measures_1d)
  touched = fill(true,length(f))
  ArrayBlock(array,touched)
end

function CellData.integrate(f::TProductGradient,a::TProductMeasure)
  fi = integrate(get_data(f),a)
  dfi = integrate(get_gradient_data(f),a)
  TProductGradientArray(fi,dfi)
end

# assembly

struct TProductSparseMatrixAssembler <: SparseMatrixAssembler
  # assem::GenericSparseMatrixAssembler
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
  a::VectorBlock{<:DomainContribution})

  array = map(collect_cell_matrix,trial.spaces_1d,test.spaces_1d,a.array)
  touched = a.touched
  ArrayBlock(array,touched)
end

function FESpaces.collect_cell_vector(
  test::TProductFESpace,
  a::VectorBlock{<:DomainContribution})

  array = map(collect_cell_vector,test.spaces_1d,a.array)
  touched = a.touched
  ArrayBlock(array,touched)
end

function FESpaces.collect_cell_matrix(
  trial::TProductFESpace,
  test::TProductFESpace,
  a::TProductGradientArray)

  f = collect_cell_matrix(trial,test,get_data(a))
  g = collect_cell_matrix(trial,test,get_gradient_data(a))
  TProductGradientArray(f,g)
end

function FESpaces.collect_cell_vector(
  test::TProductFESpace,
  a::TProductGradientArray)

  f = collect_cell_vector(test,get_data(a))
  g = collect_cell_vector(test,get_gradient_data(a))
  TProductGradientArray(f,g)
end

function FESpaces.allocate_vector(a::TProductSparseMatrixAssembler,vecdata::ArrayBlock)
  vecs_1d = map(allocate_vector,a.assems_1d,vecdata.array)
  vec = symbolic_kron(vecs_1d...)
  return TProductArray(vec,vecs_1d)
end

function FESpaces.assemble_vector!(b,a::SparseMatrixAssembler,vecdata)
  map(b.arrays_1d,assemble_vector!,a.assems_1d,vecdata.array)
  numeric_kron!(b.array,vecs_1d...)
end

function FESpaces.assemble_vector_add!(b,a::SparseMatrixAssembler,vecdata)
  map(b.arrays_1d,assemble_vector_add!,a.assems_1d,vecdata.array)
  numeric_kron!(b.array,vecs_1d...)
end

function FESpaces.assemble_vector(a::SparseMatrixAssembler,vecdata)
  vecs_1d = map(assemble_vector,a.assems_1d,vecdata.array)
  vec = kron(vecs_1d...)
  return TProductArray(vec,vecs_1d)
end

function FESpaces.allocate_matrix(a::TProductSparseMatrixAssembler,matdata::ArrayBlock)
  mats_1d = map(allocate_matrix,a.assems_1d,matdata.array)
  mat = symbolic_kron(mats_1d...)
  return TProductArray(mat,mats_1d)
end

function FESpaces.assemble_matrix!(A,a::TProductSparseMatrixAssembler,matdata)
  map(A.arrays_1d,assemble_matrix!,a.assems_1d,matdata.array)
  numeric_kron!(A.array,mats_1d...)
end

function FESpaces.assemble_matrix_add!(A,a::TProductSparseMatrixAssembler,matdata)
  map(A.arrays_1d,assemble_matrix_add!,a.assems_1d,matdata.array)
  numeric_kron!(A.array,mats_1d...)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata.array)
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

function TProductArray(arrays_1d::Vector{A}) where {T,N,A<:AbstractArray{T,N}}
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

function symbolic_kron(a::AbstractVector{T},b::AbstractVector{S}) where {T,S}
  c = Vector{promote_op(*,T,S)}(undef,length(a)*length(b))
  return c
end

@inline function numeric_kron!(c::AbstractVector,a::AbstractVector,b::AbstractVector)
  kron!(c,a,b)
end

function symbolic_kron(A::AbstractSparseMatrixCSC,B::AbstractSparseMatrixCSC)
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

@inline function numeric_kron!(C::SparseMatrixCSC,A::AbstractSparseMatrixCSC,B::AbstractSparseMatrixCSC)
  nA = size(A)
  nB = size(B)

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

function numeric_kron!(A::AbstractArray,B::AbstractArray)
  copyto!(A,B)
  A
end

function symbolic_kron(A::AbstractArray,B::AbstractArray,C::AbstractArray...)
  symbolic_kron(A,symbolic_kron(B,C...))
end

function numeric_kron!(A::AbstractArray,B::AbstractArray,C::AbstractArray...)
  numeric_kron!(A,numeric_kron!(B,C...))
end
