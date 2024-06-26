"""
    TProductSparseMatrixAssembler{D,Ti} <: SparseMatrixAssembler

Assembly-related information when constructing a [`TProductArray`](ref)

"""
struct TProductSparseMatrixAssembler{R<:TProductIndexMap,C<:TProductIndexMap} <: SparseMatrixAssembler
  assem::GenericSparseMatrixAssembler
  assems_1d::Vector{GenericSparseMatrixAssembler}
  row_index_map::R
  col_index_map::C
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::TProductFESpace,
  test::TProductFESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy())

  assem = SparseMatrixAssembler(mat,vec,trial.space,test.space,strategy)
  assems_1d = map((U,V)->SparseMatrixAssembler(mat,vec,U,V,strategy),trial.spaces_1d,test.spaces_1d)
  row_index_map = get_tp_dof_index_map(test)
  col_index_map = get_tp_dof_index_map(trial)
  TProductSparseMatrixAssembler(assem,assems_1d,row_index_map,col_index_map)
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::TrialFESpace{<:TProductFESpace},
  test::TProductFESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy())

  SparseMatrixAssembler(mat,vec,trial.space,test,strategy)
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
  return TProductArray(vec,vecs_1d,(a.row_index_map,))
end

function FESpaces.assemble_vector!(b,a::TProductSparseMatrixAssembler,vecdata::Vector)
  map(b.arrays_1d,assemble_vector!,a.assems_1d,vecdata)
  _numerical_kron!(b.array,b.arrays_1d...)
end

function FESpaces.assemble_vector_add!(b,a::TProductSparseMatrixAssembler,vecdata::Vector)
  map(b.arrays_1d,assemble_vector_add!,a.assems_1d,vecdata)
  _numerical_kron!(b.array,b.arrays_1d...)
end

function FESpaces.assemble_vector(a::TProductSparseMatrixAssembler,vecdata::Vector)
  vecs_1d = map(assemble_vector,a.assems_1d,vecdata)
  vec = _kron(vecs_1d...)
  return TProductArray(vec,vecs_1d,(a.row_index_map,))
end

function FESpaces.allocate_matrix(a::TProductSparseMatrixAssembler,matdata::Vector)
  mats_1d = map(allocate_matrix,a.assems_1d,matdata)
  mat = symbolic_kron(mats_1d...)
  return TProductArray(mat,mats_1d,(a.row_index_map,a.col_index_map))
end

function FESpaces.assemble_matrix!(A,a::TProductSparseMatrixAssembler,matdata::Vector)
  map(assemble_matrix!,A.arrays_1d,a.assems_1d,matdata)
  _numerical_kron!(A.array,A.arrays_1d...)
end

function FESpaces.assemble_matrix_add!(A,a::TProductSparseMatrixAssembler,matdata::Vector)
  map(assemble_matrix_add!,A.arrays_1d,a.assems_1d,matdata)
  _numerical_kron!(A.array,A.arrays_1d...)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata::Vector)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata)
  mat = _kron(mats_1d...)
  return TProductArray(mat,mats_1d,(a.row_index_map,a.col_index_map))
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
  _numerical_kron!(b.array,b.arrays_1d,b.gradients_1d,vecdata.op)
end

function FESpaces.assemble_vector_add!(b,a::TProductSparseMatrixAssembler,vecdata::TProductGradientEval)
  map(assemble_vector_add!,b.arrays_1d,a.assems_1d,vecdata.f)
  map(assemble_vector_add!,b.gradients_1d,a.assems_1d,vecdata.g)
  _numerical_kron!(b.array,b.arrays_1d,b.gradients_1d,vecdata.op)
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
  return TProductGradientArray(mat,mats_1d,gradmats_1d,(a.row_index_map,a.col_index_map))
end

function FESpaces.assemble_matrix!(A,a::TProductSparseMatrixAssembler,matdata::TProductGradientEval)
  map(assemble_matrix!,A.arrays_1d,a.assems_1d,matdata.f)
  map(assemble_matrix!,A.gradients_1d,a.assems_1d,matdata.g)
  _numerical_kron!(A.array,A.arrays_1d,A.gradients_1d,matdata.op)
end

function FESpaces.assemble_matrix_add!(A,a::TProductSparseMatrixAssembler,matdata::TProductGradientEval)
  map(assemble_matrix_add!,A.arrays_1d,a.assems_1d,matdata.f)
  map(assemble_matrix_add!,A.gradients_1d,a.assems_1d,matdata.g)
  _numerical_kron!(A.array,A.arrays_1d,A.gradients_1d,matdata.op)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata::TProductGradientEval)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata.f)
  gradmats_1d = map(assemble_matrix,a.assems_1d,matdata.g)
  mat = kronecker_gradients(mats_1d,gradmats_1d,matdata.op)
  return TProductGradientArray(mat,mats_1d,gradmats_1d,(a.row_index_map,a.col_index_map))
end

# multi field

function TProductBlockSparseMatrixAssembler(
  trial::MultiFieldFESpace{MS},
  test::MultiFieldFESpace{MS},
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
  ) where {NB,SB,P,MS<:BlockMultiFieldStyle{NB,SB,P}}

  @notimplementedif any(SB.!=1)
  NV = length(test.spaces)

  T = get_dof_value_type(trial)
  mat = SparseMatrixCSC{T,Int}
  vec = Vector{T}
  block_idx = CartesianIndices((NB,NB))
  block_assemblers = map(block_idx) do idx
    SparseMatrixAssembler(mat,vec,trial[idx[2]],test[idx[1]],strategy)
  end

  return MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P}(block_assemblers)
end
