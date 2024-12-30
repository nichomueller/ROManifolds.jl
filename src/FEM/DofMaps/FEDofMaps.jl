"""
    struct FEDofMap{A,B}
      sparse_dof_map::A
      dof_map::B
    end

Used to store the dof maps related to jacobian and residual in a FE
approximation problem. Fields:
- `sparse_dof_map`: dof map associated to the jacobian
- `dof_map`: dof map associated to the residual
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
