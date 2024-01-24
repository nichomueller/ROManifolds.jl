function ReferenceFEs._lagr_dof_cache(node_to_val::ParamArray,ndofs)
  map(node_to_val) do node_to_val
    ReferenceFEs._lagr_dof_cache(node_to_val,ndofs)
  end
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::ParamVector,
  node_comp_to_val::ParamVector,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  map(c,node_comp_to_val) do c,node_comp_to_val
    ReferenceFEs._evaluate_lagr_dof!(c,node_comp_to_val,node_and_comp_to_dof,ndofs,ncomps)
  end
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::ParamMatrix,
  node_pdof_comp_to_val::ParamMatrix,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  map(c,node_pdof_comp_to_val) do c,node_pdof_comp_to_val
    ReferenceFEs._evaluate_lagr_dof!(c,node_pdof_comp_to_val,node_and_comp_to_dof,ndofs,ncomps)
  end
end
