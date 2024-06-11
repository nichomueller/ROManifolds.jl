abstract type AbstractFEOperatorIndexMap{D,Ti} <: AbstractIndexMap{D,Ti} end

struct FEOperatorIndexMap{D,Ti,A,B} <: AbstractFESpaceIndexMap{D,Ti}
  matrix_map::A
  vector_map::B
end

function FEOperatorIndexMap(trial::FESpace,test::FESpace)
  trial_model = get_background_model(get_triangulation(trial))
  test_model = get_background_model(get_triangulation(test))
  @notimplementedif trial_model != test_model

  vector_map = get_dof_index_map(test)
  matrix_map = get_sparse_index_map(trial,test)
  FEOperatorIndexMap(matrix_map,vector_map)
end

function get_sparse_index_map(U::FESpace,V::FESpace)
  sparsity = get_sparsity(U,V)
  return get_nonzero_indices(sparsity)
end
