ReferenceFEs.num_components(::Type{<:AbstractArray{T}}) where T = num_components(T)

function ReferenceFEs._lagr_dof_cache(node_comp_to_val::AbstractParamVector,ndofs)
  T = eltype2(node_comp_to_val)
  L = param_length(node_comp_to_val)
  r = zeros(eltype(T),ndofs,L)
  ConsecutiveParamArray(r)
end

function ReferenceFEs._lagr_dof_cache(node_pdof_comp_to_val::AbstractParamMatrix,ndofs)
  _,npdofs = size(node_pdof_comp_to_val)
  T = eltype2(node_pdof_comp_to_val)
  L = param_length(node_pdof_comp_to_val)
  r = zeros(eltype(T),ndofs,npdofs,L)
  ConsecutiveParamArray(r)
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::AbstractVector,
  node_comp_to_val::ConsecutiveParamVector,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  setsize!(c,(ndofs,))
  r = c.array
  @inbounds for i in param_eachindex(c)
    for node in LinearIndices(node_and_comp_to_dof)
      comp_to_dof = node_and_comp_to_dof[node]
      comp_to_val = node_comp_to_val.data[node,i]
      for comp in 1:ncomps
        dof = comp_to_dof[comp]
        val = comp_to_val[comp]
        r[dof,i] = val
      end
    end
  end
  ConsecutiveParamArray(r)
end

function ReferenceFEs._evaluate_lagr_dof!(
  c::AbstractMatrix,
  node_pdof_comp_to_val::ConsecutiveParamMatrix,
  node_and_comp_to_dof,
  ndofs,
  ncomps)

  _,npdofs = size(node_pdof_comp_to_val)
  setsize!(c,(ndofs,npdofs))
  r = c.array
  @inbounds for i in param_eachindex(c)
    for node in LinearIndices(node_and_comp_to_dof)
      comp_to_dof = node_and_comp_to_dof[node]
      for pdof in 1:npdofs
        comp_to_val = node_pdof_comp_to_val.data[node,pdof,i]
        for comp in 1:ncomps
          dof = comp_to_dof[comp]
          val = comp_to_val[comp]
          r[dof,pdof,i] = val
        end
      end
    end
  end
  ConsecutiveParamArray(r)
end
