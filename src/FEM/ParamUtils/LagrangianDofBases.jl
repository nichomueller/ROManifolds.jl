ReferenceFEs.num_components(::Type{<:AbstractArray{T}}) where T = num_components(T)

function ReferenceFEs._lagr_dof_cache(node_to_val::AbstractParamArray,ndofs)
  param_array(param_data(node_to_val)) do n2v
    ReferenceFEs._lagr_dof_cache(n2v,ndofs)
  end
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::VectorOfCachedVectors,
  node_comp_to_val::AbstractParamVector,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  setsize!(c,(ndofs,))
  r = c.data.array
  for node in LinearIndices(node_and_comp_to_dof)
    comp_to_dof = node_and_comp_to_dof[node]
    comp_to_val = node_comp_to_val.data[node,:]
    for comp in 1:ncomps
      dof = comp_to_dof[comp]
      for (ip,val) in enumerate(comp_to_val)
        r[dof,ip] = val[ip][comp]
      end
    end
  end
  ArrayOfArrays(r)
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::MatrixOfCachedMatrices,
  node_pdof_comp_to_val::AbstractParamMatrix,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  _, npdofs = innersize(node_pdof_comp_to_val)
  setsize!(c,(ndofs,npdofs))
  r = c.data.array
  for node in LinearIndices(node_and_comp_to_dof)
    comp_to_dof = node_and_comp_to_dof[node]
    for pdof in 1:npdofs
      comp_to_val = node_pdof_comp_to_val.data[node,pdof,:]
      for comp in 1:ncomps
        dof = comp_to_dof[comp]
        for (ip,val) in enumerate(comp_to_val)
          r[dof,pdof,ip] = val[ip][comp]
        end
      end
    end
  end
  ArrayOfArrays(r)
end
