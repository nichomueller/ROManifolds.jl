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
function get_vector_index_map(test::SingleFieldFESpace,t::Triangulation...)
  TrivialIndexMap(num_free_dofs(test))
end

function get_vector_index_map(test::MultiFieldFESpace,t::Triangulation...)
  ntest = num_fields(test)
  index_maps = Vector{AbstractIndexMap}(undef,ntest)
  for i in 1:ntest
    index_maps[i] = get_vector_index_map(test[i],t...)
  end
  return index_maps
end

"""
    get_matrix_index_map(trial::FESpace,test::FESpace) -> AbstractIndexMap

Returns the index maps related to jacobians in a FE problem. The default output
is a TrivialIndexMap; when the trial and test spaces are of type TProductFESpace,
a SparseIndexMap is returned

"""
function get_matrix_index_map(trial::SingleFieldFESpace,test::SingleFieldFESpace,t::Triangulation...)
  sparsity = get_sparsity(trial,test,t...)
  TrivialIndexMap(sparsity)
end

function get_matrix_index_map(trial::MultiFieldFESpace,test::MultiFieldFESpace,t::Triangulation...)
  ntest = num_fields(test)
  ntrial = num_fields(trial)
  index_maps = Matrix{AbstractIndexMap}(undef,ntest,ntrial)
  for (i,j) in Iterators.product(1:ntest,1:ntrial)
    index_maps[i,j] = get_matrix_index_map(trial[j],test[i],t...)
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

function get_vector_index_map(test::FESpace,trians::Tuple{Vararg{Triangulation}})
  contribution(trians) do trian
    get_vector_index_map(test,trian)
  end
end

function get_matrix_index_map(trial::FESpace,test::FESpace,trians::Tuple{Vararg{Triangulation}})
  contribution(trians) do trian
    get_matrix_index_map(trial,test,trian)
  end
end

function get_matrix_index_map(trial::FESpace,test::FESpace,trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})
  index_maps = ()
  for t in trians
    index_maps = (index_maps...,get_matrix_index_map(trial,test,t))
  end
  return index_maps
end

function Utils.set_domains(
  i::FEOperatorIndexMap,
  trial::SingleFieldFESpace,
  test::SingleFieldFESpace,
  trians_res,
  trians_jac)

  sparsity = get_sparsity(trial,test)
  matrix_map = _get_matrix_index_map(i,trians_jac)
  vector_map = _get_vector_index_map(i,trians_res)
  matrix_map′ = sum_maps(sparsity,matrix_map)
  vector_map′ = sum_maps(vector_map)
  FEOperatorIndexMap(matrix_map′,vector_map′)
end

function Utils.set_domains(
  i::FEOperatorIndexMap,
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace,
  trians_res,
  trians_jac)

  ntest = num_fields(test)
  ntrial = num_fields(trial)
  matrix_map = _get_matrix_index_map(i,trians_jac)
  vector_map = _get_vector_index_map(i,trians_res)
  matrix_map′ = Matrix{AbstractIndexMap}(undef,ntest,ntrial)
  vector_map′ = Vector{AbstractIndexMap}(undef,ntest)
  for i in 1:ntest
    vmi = map(x -> getindex(x,i),vector_map)
    vector_map′[i] = sum_maps(vmi)
    for j in 1:ntrial
      sparsity = get_sparsity(trial[j],test[i])
      mmij = map(x -> getindex(x,i,j),matrix_map)
      matrix_map′[i,j] = sum_maps(sparsity,mmij)
    end
  end
  FEOperatorIndexMap(matrix_map′,vector_map′)
end

function Utils.change_domains(i::FEOperatorIndexMap,trial::FESpace,test::FESpace,trians_res,trians_jac)
  matrix_map′ = change_domains(get_matrix_index_map(i),trians_jac)
  vector_map′ = change_domains(get_vector_index_map(i),trians_res)
  FEOperatorIndexMap(matrix_map′,vector_map′)
end

# Utils

function _get_vector_index_map(i::FEOperatorIndexMap,trians::Tuple{Vararg{Triangulation}})
  get_values(get_vector_index_map(i))
end

function _get_matrix_index_map(i::FEOperatorIndexMap,trians::Tuple{Vararg{Triangulation}})
  get_values(get_matrix_index_map(i))
end

function _get_matrix_index_map(i::FEOperatorIndexMap,trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})
  _matrix_map = get_values(get_matrix_index_map(i))
  matrix_map = ()
  for _imap in _matrix_map
    matrix_map = (matrix_map...,_imap...)
  end
  return matrix_map
end
