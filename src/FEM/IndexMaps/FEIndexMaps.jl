"""
    struct FEOperatorDofMap{A,B}
      matrix_map::A
      vector_map::B
    end

Used to store the index maps related to jacobians and residuals in a FE problem

"""
struct FEOperatorDofMap{A,B}
  matrix_map::A
  vector_map::B
end

get_matrix_dof_map(i::FEOperatorDofMap) = i.matrix_map
get_vector_dof_map(i::FEOperatorDofMap) = i.vector_map

function FEOperatorDofMap(trial::FESpace,test::FESpace)
  trial_model = get_background_model(get_triangulation(trial))
  test_model = get_background_model(get_triangulation(test))
  @notimplementedif trial_model != test_model

  vector_map = get_vector_dof_map(test)
  matrix_map = get_matrix_dof_map(trial,test)
  FEOperatorDofMap(matrix_map,vector_map)
end

"""
    get_vector_dof_map(test::FESpace) -> AbstractDofMap

Returns the index maps related to residuals in a FE problem. The default output
is a TrivialDofMap; when the test space is of type TProductFESpace, a
nontrivial index map is returned

"""
function get_vector_dof_map(test::SingleFieldFESpace,t::Triangulation...)
  TrivialDofMap(num_free_dofs(test))
end

function get_vector_dof_map(test::MultiFieldFESpace,t::Triangulation...)
  ntest = num_fields(test)
  dof_maps = Vector{AbstractDofMap}(undef,ntest)
  for i in 1:ntest
    dof_maps[i] = get_vector_dof_map(test[i],t...)
  end
  return dof_maps
end

"""
    get_matrix_dof_map(trial::FESpace,test::FESpace) -> AbstractDofMap

Returns the index maps related to jacobians in a FE problem. The default output
is a TrivialDofMap; when the trial and test spaces are of type TProductFESpace,
a SparseDofMap is returned

"""
function get_matrix_dof_map(trial::SingleFieldFESpace,test::SingleFieldFESpace)
  sparsity = SparsityPattern(trial,test)
  TrivialDofMap(sparsity)
end

function get_matrix_dof_map(trial::MultiFieldFESpace,test::MultiFieldFESpace,t::Triangulation...)
  ntest = num_fields(test)
  ntrial = num_fields(trial)
  dof_maps = Matrix{AbstractDofMap}(undef,ntest,ntrial)
  for (i,j) in Iterators.product(1:ntest,1:ntrial)
    dof_maps[i,j] = get_matrix_dof_map(trial[j],test[i],t...)
  end
  return dof_maps
end

# triangulation utils

function FEOperatorDofMap(trial::FESpace,test::FESpace,trians_res,trians_jacs)
  trial_model = get_background_model(get_triangulation(trial))
  test_model = get_background_model(get_triangulation(test))
  @notimplementedif trial_model != test_model

  vector_map = get_vector_dof_map(test,trians_res)
  matrix_map = get_matrix_dof_map(trial,test,trians_jacs)
  FEOperatorDofMap(matrix_map,vector_map)
end

function get_vector_dof_map(test::FESpace,trians::Tuple{Vararg{Triangulation}})
  contribution(trians) do trian
    get_vector_dof_map(test,trian)
  end
end

function get_matrix_dof_map(trial::FESpace,test::FESpace,trians::Tuple{Vararg{Triangulation}})
  contribution(trians) do trian
    get_matrix_dof_map(trial,test,trian)
  end
end

function get_matrix_dof_map(trial::FESpace,test::FESpace,trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})
  dof_maps = ()
  for t in trians
    dof_maps = (dof_maps...,get_matrix_dof_map(trial,test,t))
  end
  return dof_maps
end

function Utils.set_domains(
  i::FEOperatorDofMap,
  trial::SingleFieldFESpace,
  test::SingleFieldFESpace,
  trians_res,
  trians_jac)

  sparsity = get_sparsity(trial,test)
  matrix_map = _get_matrix_dof_map(i,trians_jac)
  vector_map = _get_vector_dof_map(i,trians_res)
  matrix_map′ = sum_maps(sparsity,matrix_map)
  vector_map′ = sum_maps(vector_map)
  FEOperatorDofMap(matrix_map′,vector_map′)
end

function Utils.set_domains(
  i::FEOperatorDofMap,
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace,
  trians_res,
  trians_jac)

  ntest = num_fields(test)
  ntrial = num_fields(trial)
  matrix_map = _get_matrix_dof_map(i,trians_jac)
  vector_map = _get_vector_dof_map(i,trians_res)
  matrix_map′ = Matrix{AbstractDofMap}(undef,ntest,ntrial)
  vector_map′ = Vector{AbstractDofMap}(undef,ntest)
  for i in 1:ntest
    vmi = map(x -> getindex(x,i),vector_map)
    vector_map′[i] = sum_maps(vmi)
    for j in 1:ntrial
      sparsity = get_sparsity(trial[j],test[i])
      mmij = map(x -> getindex(x,i,j),matrix_map)
      matrix_map′[i,j] = sum_maps(sparsity,mmij)
    end
  end
  FEOperatorDofMap(matrix_map′,vector_map′)
end

function Utils.change_domains(i::FEOperatorDofMap,trial::FESpace,test::FESpace,trians_res,trians_jac)
  matrix_map′ = change_domains(get_matrix_dof_map(i),trians_jac)
  vector_map′ = change_domains(get_vector_dof_map(i),trians_res)
  FEOperatorDofMap(matrix_map′,vector_map′)
end

# Utils

function _get_vector_dof_map(i::FEOperatorDofMap,trians::Tuple{Vararg{Triangulation}})
  get_values(get_vector_dof_map(i))
end

function _get_matrix_dof_map(i::FEOperatorDofMap,trians::Tuple{Vararg{Triangulation}})
  get_values(get_matrix_dof_map(i))
end

function _get_matrix_dof_map(i::FEOperatorDofMap,trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})
  _matrix_map = get_values(get_matrix_dof_map(i))
  matrix_map = ()
  for _imap in _matrix_map
    matrix_map = (matrix_map...,_imap...)
  end
  return matrix_map
end
