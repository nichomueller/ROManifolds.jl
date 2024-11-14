"""
    TProductSparseMatrixAssembler{A<:SparseMatrixAssembler,R,C} <: SparseMatrixAssembler

Assembly-related information when constructing a [`AbstractRankTensor`](ref)

"""
struct TProductSparseMatrixAssembler{A<:SparseMatrixAssembler,R,C} <: SparseMatrixAssembler
  assems_1d::Vector{A}
  row_dof_map::R
  col_dof_map::C
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::TProductFESpace,
  test::TProductFESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy())

  assems_1d = map((U,V)->SparseMatrixAssembler(mat,vec,U,V,strategy),trial.spaces_1d,test.spaces_1d)
  row_dof_map = get_tp_dof_dof_map(test)
  col_dof_map = get_tp_dof_dof_map(trial)
  TProductSparseMatrixAssembler(assems_1d,row_dof_map,col_dof_map)
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
  a::GenericTProductDiffEval)

  f = collect_cell_matrix(trial,test,get_data(a))
  g = collect_cell_matrix(trial,test,get_diff_data(a))
  GenericTProductDiffEval(a.op,f,g,a.summation)
end

function FESpaces.collect_cell_vector(
  test::FESpace,
  a::GenericTProductDiffEval)

  f = collect_cell_vector(test,get_data(a))
  g = collect_cell_vector(test,get_diff_data(a))
  GenericTProductDiffEval(a.op,f,g,a.summation)
end

function FESpaces.allocate_vector(a::TProductSparseMatrixAssembler,vecdata::Vector)
  vecs_1d = map(allocate_vector,a.assems_1d,vecdata)
  return tproduct_array(vecs_1d,(a.row_dof_map,))
end

function FESpaces.assemble_vector!(b,a::TProductSparseMatrixAssembler,vecdata::Vector)
  map(b.arrays_1d,assemble_vector!,a.assems_1d,vecdata)
end

function FESpaces.assemble_vector_add!(b,a::TProductSparseMatrixAssembler,vecdata::Vector)
  map(b.arrays_1d,assemble_vector_add!,a.assems_1d,vecdata)
end

function FESpaces.assemble_vector(a::TProductSparseMatrixAssembler,vecdata::Vector)
  vecs_1d = map(assemble_vector,a.assems_1d,vecdata)
  return tproduct_array(vecs_1d,(a.row_dof_map,))
end

function FESpaces.allocate_matrix(a::TProductSparseMatrixAssembler,matdata::Vector)
  mats_1d = map(allocate_matrix,a.assems_1d,matdata)
  return tproduct_array(mats_1d,(a.row_dof_map,a.col_dof_map))
end

function FESpaces.assemble_matrix!(A,a::TProductSparseMatrixAssembler,matdata::Vector)
  map(assemble_matrix!,A.arrays_1d,a.assems_1d,matdata)
end

function FESpaces.assemble_matrix_add!(A,a::TProductSparseMatrixAssembler,matdata::Vector)
  map(assemble_matrix_add!,A.arrays_1d,a.assems_1d,matdata)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata::Vector)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata)
  return tproduct_array(mats_1d,(a.row_dof_map,a.col_dof_map))
end

function FESpaces.allocate_vector(a::TProductSparseMatrixAssembler,vecdata::GenericTProductDiffEval)
  vecs_1d = map(allocate_vector,a.assems_1d,vecdata.f)
  gradvecs_1d = map(allocate_vector,a.assems_1d,vecdata.g)
  return tproduct_array(vecdata.op,vecs_1d,gradvecs_1d,(a.row_dof_map,),vecdata.summation)
end

function FESpaces.assemble_vector!(b,a::TProductSparseMatrixAssembler,vecdata::GenericTProductDiffEval)
  map(assemble_vector!,b.arrays_1d,a.assems_1d,vecdata.f)
  map(assemble_vector!,b.gradients_1d,a.assems_1d,vecdata.g)
end

function FESpaces.assemble_vector_add!(b,a::TProductSparseMatrixAssembler,vecdata::GenericTProductDiffEval)
  map(assemble_vector_add!,b.arrays_1d,a.assems_1d,vecdata.f)
  map(assemble_vector_add!,b.gradients_1d,a.assems_1d,vecdata.g)
end

function FESpaces.assemble_vector(a::TProductSparseMatrixAssembler,vecdata::GenericTProductDiffEval)
  vecs_1d = map(assemble_vector,a.assems_1d,vecdata.f)
  gradvecs_1d = map(assemble_vector,a.assems_1d,vecdata.g)
  return tproduct_array(vecdata.op,vecs_1d,gradvecs_1d,(a.row_dof_map,),vecdata.summation)
end

function FESpaces.allocate_matrix(a::TProductSparseMatrixAssembler,matdata::GenericTProductDiffEval)
  mats_1d = map(allocate_matrix,a.assems_1d,matdata.f)
  gradmats_1d = map(allocate_matrix,a.assems_1d,matdata.g)
  return tproduct_array(matdata.op,mats_1d,gradmats_1d,(a.row_dof_map,a.col_dof_map),matdata.summation)
end

function FESpaces.assemble_matrix!(A,a::TProductSparseMatrixAssembler,matdata::GenericTProductDiffEval)
  map(assemble_matrix!,A.arrays_1d,a.assems_1d,matdata.f)
  map(assemble_matrix!,A.gradients_1d,a.assems_1d,matdata.g)
end

function FESpaces.assemble_matrix_add!(A,a::TProductSparseMatrixAssembler,matdata::GenericTProductDiffEval)
  map(assemble_matrix_add!,A.arrays_1d,a.assems_1d,matdata.f)
  map(assemble_matrix_add!,A.gradients_1d,a.assems_1d,matdata.g)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata::GenericTProductDiffEval)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata.f)
  gradmats_1d = map(assemble_matrix,a.assems_1d,matdata.g)
  return tproduct_array(matdata.op,mats_1d,gradmats_1d,(a.row_dof_map,a.col_dof_map),matdata.summation)
end

# multi field

function TProductBlockSparseMatrixAssembler(trial::MultiFieldFESpace,test::MultiFieldFESpace)
  assems_1d = map(eachindex(test.spaces[1].spaces_1d)) do d
    trials_d = map(f->_remove_trial(f).spaces_1d[d],trial.spaces)
    tests_d = map(f->f.spaces_1d[d],test.spaces)
    trial′ = MultiFieldFESpace(trial.vector_type,trials_d,trial.multi_field_style)
    test′ = MultiFieldFESpace(test.vector_type,tests_d,test.multi_field_style)
    SparseMatrixAssembler(trial′,test′)
  end
  row_dof_map = map(get_tp_dof_dof_map,test.spaces)
  col_dof_map = map(get_tp_dof_dof_map,_remove_trial.(trial.spaces))
  TProductSparseMatrixAssembler(assems_1d,row_dof_map,col_dof_map)
end
