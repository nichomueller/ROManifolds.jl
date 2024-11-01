function get_dof_index_map(model::CartesianDiscreteModel,space::FESpaceWithLinearConstraints)
  index_map = get_dof_index_map(model,space.space)
end
