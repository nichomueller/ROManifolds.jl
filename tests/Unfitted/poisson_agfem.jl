function cutfem_dof_to_tag(Vbg::FESpace,Ωact_in::Triangulation,Ωact_out::Triangulation)
  tags,tag_to_ndofs,dof_to_tag = my_get_dof_to_tag(Vbg,Ωact_in,Ωact_out)
  dof_to_pdof = Utils.get_dof_to_colored_dof(tag_to_ndofs,dof_to_tag)
  MultiColorFESpace(Vbg,tags,tag_to_ndofs,dof_to_tag,dof_to_pdof)
end

function my_get_dof_to_tag(space::FESpace,Ωact_in::Triangulation,Ωact_out::Triangulation)
  tags = ["in","out","interface"]
  ntags = length(tags)
  ndofs = num_free_dofs(space)

  reject_ndofs = ndofs
  tag_to_ndofs = Vector{Int64}(undef,ntags)
  dof_to_tag   = fill(Int8(-1),ndofs)

  dof_to_mask_in = my_get_dof_mask(space,Ωact_in)
  dof_to_mask_out = my_get_dof_mask(space,Ωact_out)
  dof_to_mask_Γ = dof_to_mask_in .&& dof_to_mask_out
  dof_to_mask_in[findall(dof_to_mask_Γ)] .= false
  dof_to_mask_out[findall(dof_to_mask_Γ)] .= false

  for (i,dof_to_mask) in enumerate((dof_to_mask_in,dof_to_mask_out,dof_to_mask_Γ))
    dof_to_tag[dof_to_mask] .= i
    i_ndofs = sum(dof_to_mask)
    tag_to_ndofs[i] = i_ndofs
    reject_ndofs   -= i_ndofs
  end
  @assert sum(tag_to_ndofs) == (ndofs-reject_ndofs) "There is overlapping between the tags!"
  return tags,tag_to_ndofs,dof_to_tag
end

function my_get_dof_mask(space::FESpace,trian::Triangulation{Dc}) where Dc
  tface_to_mface = get_tface_to_mface(trian)

  cell_dof_ids = get_cell_dof_ids(space)

  cell_dof_ids_cache = array_cache(cell_dof_ids)
  dof_to_mask = fill(false,num_free_dofs(space))
  for cell in tface_to_mface
    dofs = getindex!(cell_dof_ids_cache,cell_dof_ids,cell)
    for dof in dofs
      if dof > 0
        dof_to_mask[dof] = true
      end
    end
  end

  return dof_to_mask
end
