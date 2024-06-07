function ReferenceFEs._lagr_dof_cache(node_to_val::AbstractParamArray,ndofs)
  param_array(param_data(node_to_val)) do n2v
    ReferenceFEs._lagr_dof_cache(n2v,ndofs)
  end
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::AbstractParamVector,
  node_comp_to_val::AbstractParamVector,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  @inbounds for i = param_eachindex(node_to_val)
    c_i = param_view(c,i)
    node_comp_to_val_i = param_view(node_comp_to_val,i)
    ReferenceFEs._evaluate_lagr_dof!(c_i,node_comp_to_val_i,node_and_comp_to_dof,ndofs,ncomps)
  end
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::AbstractParamMatrix,
  node_pdof_comp_to_val::AbstractParamMatrix,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  @inbounds for i = param_eachindex(node_to_val)
    c_i = param_view(c,i)
    node_pdof_comp_to_val_i = param_view(node_pdof_comp_to_val,i)
    ReferenceFEs._evaluate_lagr_dof!(c_i,node_pdof_comp_to_val_i,node_and_comp_to_dof,ndofs,ncomps)
  end
end
