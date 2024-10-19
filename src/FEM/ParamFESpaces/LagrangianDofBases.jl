ReferenceFEs.num_components(::Type{<:AbstractArray{T}}) where T = num_components(T)

function ReferenceFEs._lagr_dof_cache(node_to_val::AbstractParamArray,ndofs)
  map(get_param_data(node_to_val)) do n2v
    ReferenceFEs._lagr_dof_cache(n2v,ndofs)
  end |> ParamArray
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::AbstractParamVector,
  node_comp_to_val::AbstractParamVector,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  param_array(get_param_data(c),get_param_data(node_comp_to_val)) do c,node_comp_to_val
    ReferenceFEs._evaluate_lagr_dof!(c,node_comp_to_val,node_and_comp_to_dof,ndofs,ncomps)
  end
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::AbstractParamMatrix,
  node_pdof_comp_to_val::AbstractParamMatrix,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  param_array(get_param_data(c),get_param_data(node_pdof_comp_to_val)) do c,node_pdof_comp_to_val
    ReferenceFEs._evaluate_lagr_dof!(c,node_pdof_comp_to_val,node_and_comp_to_dof,ndofs,ncomps)
  end
end
