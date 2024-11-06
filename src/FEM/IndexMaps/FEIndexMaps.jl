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

function get_vector_index_map(test::MultiFieldFESpace)
  ntest = num_fields(test)
  index_maps = Vector{AbstractIndexMap}(undef,ntest)
  for i in 1:ntest
    index_maps[i] = get_vector_index_map(test[i])
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

function get_matrix_index_map(trial::MultiFieldFESpace,test::MultiFieldFESpace)
  ntest = num_fields(test)
  ntrial = num_fields(trial)
  index_maps = Matrix{AbstractIndexMap}(undef,ntest,ntrial)
  for (i,j) in Iterators.product(1:ntest,1:ntrial)
    index_maps[i,j] = get_matrix_index_map(trial[j],test[i])
  end
  return index_maps
end

# triangulation utils

function FEOperatorIndexMap(trial::FESpace,test::FESpace,trians_res,trians_jacs)
  trial_model = get_background_model(get_triangulation(trial))
  test_model = get_background_model(get_triangulation(test))
  @notimplementedif trial_model != test_model

  vector_map = get_vector_index_map(test,trians_res)
  matrix_map = get_matrix_index_map(trial,test,trians_jacs)
  FEOperatorIndexMap(matrix_map,vector_map)
end

function get_vector_index_map(
  test::FESpace,
  trians::Tuple{Vararg{Triangulation}})

  contribution(trians) do trian
    get_vector_index_map(test)
  end
end

function get_matrix_index_map(
  trial::FESpace,
  test::FESpace,
  trians::Tuple{Vararg{Triangulation}})

  contribution(trians) do trian
    sparsity = get_sparsity(trial,test,trian)
    TrivialIndexMap(sparsity)
  end
end

function get_matrix_index_map(
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace,
  trians::Tuple{Vararg{Triangulation}})

  ntest = num_fields(test)
  ntrial = num_fields(trial)
  contribution(trians) do trian
    index_maps = Matrix{AbstractIndexMap}(undef,ntest,ntrial)
    for (i,j) in Iterators.product(1:ntest,1:ntrial)
      index_maps[i,j] = get_matrix_index_map(trial[j],test[i],trian)
    end
    index_maps
  end
end

function get_matrix_index_map(
  trial::FESpace,
  test::FESpace,
  trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})

  index_maps = ()
  for t in trians
    index_maps = (index_maps...,get_matrix_index_map(trial,test,t))
  end
  return index_maps
end

function Utils.set_domains(i::FEOperatorIndexMap,trians_res,trians_jac)
  matrix_map′ = sum_maps(get_values(get_matrix_index_map(i))...)
  vector_map′ = sum_maps(get_values(get_vector_index_map(i))...)
  FEOperatorIndexMap(matrix_map′,vector_map′)
end

function Utils.change_domains(i::FEOperatorIndexMap,trians_res,trians_jac)
  matrix_map′ = change_domains(get_matrix_index_map(i),trians_jac)
  vector_map′ = change_domains(get_vector_index_map(i),trians_res)
  FEOperatorIndexMap(matrix_map′,vector_map′)
end
