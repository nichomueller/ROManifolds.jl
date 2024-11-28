"""
    struct FEDofMap{A,B}
      sparse_dof_map::A
      dof_map::B
    end

Used to store the index maps related to jacobians and residuals in a FE problem

"""
struct FEDofMap{A,B}
  sparse_dof_map::A
  dof_map::B
end

get_sparse_dof_map(i::FEDofMap) = i.sparse_dof_map
get_dof_map(i::FEDofMap) = i.dof_map

function FEDofMap(trial::FESpace,test::FESpace)
  trial_model = get_background_model(get_triangulation(trial))
  test_model = get_background_model(get_triangulation(test))
  @notimplementedif trial_model != test_model

  dof_map = get_dof_map(test)
  sparse_dof_map = get_sparse_dof_map(trial,test)
  FEDofMap(sparse_dof_map,dof_map)
end
