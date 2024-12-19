struct NewFESpace{A} <: SingleFieldFESpace
  space::A
  cell_dof_ids::AbstractArray
end

function NewFESpace(space::FESpace,trian_in::Triangulation)
  cell_dof_ids = change_cell_dof_ids(space,trian_in)
  NewFESpace(space,cell_dof_ids)
end

# FESpace interface

FESpaces.get_fe_basis(W::NewFESpace) = get_fe_basis(W.space)
FESpaces.get_trial_fe_basis(W::NewFESpace) = get_trial_fe_basis(W.space)
FESpaces.ConstraintStyle(W::NewFESpace) = ConstraintStyle(W.space)
CellData.get_triangulation(W::NewFESpace) = get_triangulation(W.space)
FESpaces.get_fe_dof_basis(W::NewFESpace) = get_fe_dof_basis(W.space)
FESpaces.ConstraintStyle(::Type{<:NewFESpace{A}}) where A = ConstraintStyle(A)
FESpaces.get_vector_type(W::NewFESpace) = get_vector_type(W.space)
FESpaces.get_dof_value_type(W::NewFESpace) = get_dof_value_type(W.space)
FESpaces.get_free_dof_ids(W::NewFESpace) = get_free_dof_ids(W.space)
FESpaces.zero_free_values(W::NewFESpace) = zero_free_values(W.space)
FESpaces.get_dirichlet_dof_ids(W::NewFESpace) = get_dirichlet_dof_ids(W.space)
FESpaces.get_dirichlet_dof_tag(W::NewFESpace) = get_dirichlet_dof_tag(W.space)
FESpaces.num_dirichlet_tags(W::NewFESpace) = num_dirichlet_tags(W.space)
FESpaces.get_cell_dof_ids(W::NewFESpace) = W.cell_dof_ids

function FESpaces.scatter_free_and_dirichlet_values(W::NewFESpace,fv,dv)
  scatter_free_and_dirichlet_values(W.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,W::NewFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,W.space,cv)
end

# utils

const ACTIVE_COLOR = Int8(1)
const INTERFACE_COLOR = Int8(2)
const INACTIVE_COLOR = Int8(3)

function change_cell_dof_ids(space::FESpace,trian_in::Triangulation)
  trian = get_triangulation(space)
  trian_out = complementary(trian_in,trian)
  dof_to_color = get_dof_to_color(space,trian_in,trian_out)

  cell_dof_ids = copy(get_cell_dof_ids(space))
  cells_out = get_tface_to_mface(trian_out)
  for cell in cells_out
    change_cell_dof_ids!(cell_dof_ids,dof_to_color,cell)
  end

  return cell_dof_ids
end

function change_cell_dof_ids!(cell_dof_ids::Table,dof_to_color::AbstractVector,i::Int)
  pini = cell_dof_ids.ptrs[i]
  pend = cell_dof_ids.ptrs[i+1]-1
  for p in pini:pend
    dof = cell_dof_ids.data[p]
    if dof > 0
      color = dof_to_color[dof]
      if color == INTERFACE_COLOR
        cell_dof_ids.data[p] = 0
      end
    end
  end
end

function get_dof_to_color(space::FESpace,trian_in::Triangulation,trian_out::Triangulation)
  ncolors = 3 # in, interface, out
  ndofs = num_free_dofs(space)

  reject_ndofs = ndofs
  color_to_ndofs = Vector{Int64}(undef,ncolors)
  dof_to_color   = fill(Int8(-1),ndofs)

  dof_to_mask_in = get_dof_mask(space,trian_in)
  dof_to_mask_out = get_dof_mask(space,trian_out)
  dof_to_mask_Γ = dof_to_mask_in .&& dof_to_mask_out
  dof_to_mask_in[findall(dof_to_mask_Γ)] .= false
  dof_to_mask_out[findall(dof_to_mask_Γ)] .= false

  for (i,dof_to_mask) in enumerate((dof_to_mask_in,dof_to_mask_Γ,dof_to_mask_out))
    dof_to_color[dof_to_mask] .= i
    i_ndofs = sum(dof_to_mask)
    color_to_ndofs[i] = i_ndofs
    reject_ndofs -= i_ndofs
  end
  @assert sum(color_to_ndofs) == (ndofs-reject_ndofs) "There is overlapping between the colors!"

  return dof_to_color
end

function get_dof_mask(space::FESpace,trian::Triangulation{Dc}) where Dc
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

function get_cell_to_color(trian::Triangulation,trian_in::Triangulation,trian_out::Triangulation)
  cell_to_color = fill(Int8(-1),num_cells(trian))
  cell_to_mask_in = get_cell_mask(trian,trian_in)
  cell_to_mask_out = get_cell_mask(trian,trian_out)
  cell_to_mask_Γ = cell_to_mask_in .&& cell_to_mask_out
  cell_to_mask_in[findall(cell_to_mask_Γ)] .= false
  cell_to_mask_out[findall(cell_to_mask_Γ)] .= false
  for (i,cell_to_mask) in enumerate((cell_to_mask_in,cell_to_mask_Γ,cell_to_mask_out))
    cell_to_color[cell_to_mask] .= i
  end
  return cell_to_color
end

function get_cell_mask(trian::Triangulation,t::Triangulation)
  @check is_included(t,trian)
  cell_to_mask = fill(false,num_cells(trian))
  tface_to_mface = get_tface_to_mface(t)
  cell_to_mask[tface_to_mface] .= true
  return cell_to_mask
end

function complementary(trian_in::Triangulation,trian::Triangulation)
  tface_to_mface = get_tface_to_mface(trian)
  tface_in_to_mface = get_tface_to_mface(trian_in)
  tface_out_to_mface = setdiff(tface_to_mface,tface_in_to_mface)
  view(trian,tface_out_to_mface)
end
