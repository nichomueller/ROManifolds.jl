"""
    struct FEOperatorIndexMap{A,B}
      matrix_map::A
      vector_map::B
    end

Used to store the index maps related to jacobians and residuals in a FE problem

"""
struct FEOperatorIndexMap{A,B}
  matrix_map::A
  vector_map::B
end

get_matrix_index_map(i::FEOperatorIndexMap) = i.matrix_map
get_vector_index_map(i::FEOperatorIndexMap) = i.vector_map

function FEOperatorIndexMap(trial::FESpace,test::FESpace)
  trial_model = get_background_model(get_triangulation(trial))
  test_model = get_background_model(get_triangulation(test))
  @notimplementedif trial_model != test_model

  vector_map = get_vector_index_map(test)
  matrix_map = get_matrix_index_map(trial,test)
  FEOperatorIndexMap(matrix_map,vector_map)
end

"""
    get_vector_index_map(test::FESpace) -> AbstractIndexMap

Returns the index maps related to residuals in a FE problem. The default output
is a TrivialIndexMap; when the test space is of type TProductFESpace, a
nontrivial index map is returned

"""
function get_vector_index_map(test::FESpace)
  TrivialIndexMap(num_free_dofs(test))
end

function get_vector_index_map(tests::MultiFieldFESpace)
  ntests = num_fields(tests)
  index_maps = Vector{AbstractIndexMap}(undef,ntests)
  for i in 1:ntests
    index_maps[i] = get_vector_index_map(tests[i])
  end
  return index_maps
end


"""
    get_matrix_index_map(trial::FESpace,test::FESpace) -> AbstractIndexMap

Returns the index maps related to jacobians in a FE problem. The default output
is a TrivialIndexMap; when the trial and test spaces are of type TProductFESpace,
a SparseIndexMap is returned

"""
function get_matrix_index_map(trial::FESpace,test::FESpace)
  sparsity = get_sparsity(trial,test)
  TrivialIndexMap(sparsity)
end

for F in (:TrialFESpace,:TransientTrialFESpace)
  @eval begin
    get_matrix_index_map(trial::$F,tests::SingleFieldFESpace) = get_matrix_index_map(trial.space,tests)
  end
end

function get_matrix_index_map(trials::MultiFieldFESpace,tests::MultiFieldFESpace)
  ntests = num_fields(tests)
  ntrials = num_fields(trials)
  index_maps = Matrix{AbstractIndexMap}(undef,ntests,ntrials)
  for (i,j) in Iterators.product(1:ntests,1:ntrials)
    index_maps[i,j] = get_matrix_index_map(trials[j],tests[i])
  end
  return index_maps
end
