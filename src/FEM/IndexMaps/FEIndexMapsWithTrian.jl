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
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace,
  trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})

  index_maps = ()
  for t in trians
    index_maps = (index_maps...,get_matrix_index_map(trial,test,trians))
  end
  return index_maps
end
