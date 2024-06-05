function ReferenceFEs._lagr_dof_cache(node_to_val::AbstractParamArray,ndofs)
  ParamArray([
    ReferenceFEs._lagr_dof_cache(param_getindex(node_to_val,i),ndofs)
    for i = param_eachindex(node_to_val)
    ])
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::AbstractParamVector,
  node_comp_to_val::AbstractParamVector,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  @inbounds for i = param_eachindex(node_to_val)
    c_i = param_getindex(c,i)
    node_comp_to_val_i = param_getindex(node_comp_to_val,i)
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
    c_i = param_getindex(c,i)
    node_pdof_comp_to_val_i = param_getindex(node_pdof_comp_to_val,i)
    ReferenceFEs._evaluate_lagr_dof!(c_i,node_pdof_comp_to_val_i,node_and_comp_to_dof,ndofs,ncomps)
  end
end
