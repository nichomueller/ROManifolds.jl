"""
    TProductSparseMatrixAssembler{A<:SparseMatrixAssemblerR,C} <: SparseMatrixAssembler

Assembly-related information when constructing a [`TProductArray`](ref)

"""
struct TProductSparseMatrixAssembler{A<:SparseMatrixAssembler,R,C} <: SparseMatrixAssembler
  assem::A
  assems_1d::Vector{A}
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
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace,
  a::Vector{<:DomainContribution})

  map(eachindex(a)) do d
    trials_d = map(f->_remove_trial(f).spaces_1d[d],trial.spaces)
    tests_d = map(f->f.spaces_1d[d],test.spaces)
    trial′ = MultiFieldFESpace(trial.vector_type,trials_d,trial.multi_field_style)
    test′ = MultiFieldFESpace(test.vector_type,tests_d,test.multi_field_style)
    collect_cell_matrix(trial′,test′,a[d])
  end
end

function FESpaces.collect_cell_vector(
  test::MultiFieldFESpace,
  a::Vector{<:DomainContribution})

  map(eachindex(a)) do d
    tests_d = map(f->f.spaces_1d[d],test.spaces)
    test′ = MultiFieldFESpace(test.vector_type,tests_d,test.multi_field_style)
    collect_cell_vector(test′,a[d])
  end
end

function FESpaces.collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::TProductGradientEval)

  f = collect_cell_matrix(trial,test,get_tp_data(a))
  g = collect_cell_matrix(trial,test,get_tp_gradient_data(a))
  TProductGradientEval(f,g,a.op)
end

function FESpaces.collect_cell_vector(
  test::FESpace,
  a::TProductGradientEval)

  f = collect_cell_vector(test,get_tp_data(a))
  g = collect_cell_vector(test,get_tp_gradient_data(a))
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

function TProductBlockSparseMatrixAssembler(trial::MultiFieldFESpace,test::MultiFieldFESpace)
  assem = SparseMatrixAssembler(trial,test)
  assems_1d = map(eachindex(test.spaces[1].spaces_1d)) do d
    trials_d = map(f->_remove_trial(f).spaces_1d[d],trial.spaces)
    tests_d = map(f->f.spaces_1d[d],test.spaces)
    trial′ = MultiFieldFESpace(trial.vector_type,trials_d,trial.multi_field_style)
    test′ = MultiFieldFESpace(test.vector_type,tests_d,test.multi_field_style)
    SparseMatrixAssembler(trial′,test′)
  end
  row_index_map = map(get_tp_dof_index_map,test.spaces)
  col_index_map = map(get_tp_dof_index_map,_remove_trial.(trial.spaces))
  TProductSparseMatrixAssembler(assem,assems_1d,row_index_map,col_index_map)
end
