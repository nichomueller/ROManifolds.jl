function ReferenceFEs._lagr_dof_cache(node_to_val::PArray,ndofs)
  map(node_to_val) do node_to_val
    ReferenceFEs._lagr_dof_cache(node_to_val,ndofs)
  end
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::PArray{<:AbstractVector},
  node_comp_to_val::PArray{<:AbstractVector},
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  map(c,node_comp_to_val) do c,node_comp_to_val
    ReferenceFEs._evaluate_lagr_dof!(c,node_comp_to_val,node_and_comp_to_dof,ndofs,ncomps)
  end
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::PArray{<:AbstractMatrix},
  node_pdof_comp_to_val::PArray{<:AbstractMatrix},
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  map(c,node_pdof_comp_to_val) do c,node_pdof_comp_to_val
    ReferenceFEs._evaluate_lagr_dof!(c,node_pdof_comp_to_val,node_and_comp_to_dof,ndofs,ncomps)
  end
end
