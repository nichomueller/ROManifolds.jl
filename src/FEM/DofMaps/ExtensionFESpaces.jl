function OrderedFESpace(act_trian::Triangulation,args...;kwargs...)
  act_model = get_active_model(act_trian)
  out_trian =
  bg_model = get_background_model(trian)
  OrderedFESpace(act_model,bg_model,trian,args...;kwargs...)
end

function OrderedFESpace(
  act_model::DiscreteModel,
  bg_model::CartesianDiscreteModel,
  act_trian::Triangulation,
  args...;kwargs...)

  act_f = CartesianFESpace(act_model,args...;trian=act_trian,kwargs...)
  bg_f = CartesianFESpace(bg_model,args...;kwargs...)
  act_odofs = _get_bg_odof_to_act_odof(bg_f,act_f)
  CartesianFESpace(act_f.space,act_f.cell_odofs_ids,act_odofs,in_odofs)
end

struct ExtensionFESpace{S<:SingleFieldFESpace} <: OrderedFESpace{S}
  space::S
  bg_dofs_to_ext_dofs::AbstractVector
end

# FESpace interface

FESpaces.get_fe_space(f::ExtensionFESpace) = f.space

FESpaces.ConstraintStyle(::Type{ExtensionFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_triangulation(f::ExtensionFESpace) = get_triangulation(get_fe_space(f))

FESpaces.get_dof_value_type(f::ExtensionFESpace) = get_dof_value_type(get_fe_space(f))

FESpaces.get_fe_basis(f::ExtensionFESpace) = get_fe_basis(get_fe_space(f))

FESpaces.get_trial_fe_basis(f::ExtensionFESpace) = get_trial_fe_basis(get_fe_space(f))

FESpaces.get_fe_dof_basis(f::ExtensionFESpace) = get_fe_dof_basis(get_fe_space(f))

function FESpaces.get_free_dof_ids(f::ExtensionFESpace)
  nfree = num_free_dofs(get_fe_space(f))+length(f.bg_dofs_to_ext_dofs)
  Base.OneTo(nfree)
end

FESpaces.get_dirichlet_dof_ids(f::ExtensionFESpace) = get_dirichlet_dof_ids(get_fe_space(f))

FESpaces.get_cell_dof_ids(f::ExtensionFESpace) = f.cell_to_ladof_to_adof

FESpaces.get_cell_isconstrained(f::ExtensionFESpace) = get_cell_isconstrained(get_fe_space(f))

function FESpaces.get_cell_constraints(f::ExtensionFESpace)
  get_cell_constraints(get_fe_space(f))
end

FESpaces.get_cell_is_dirichlet(f::ExtensionFESpace) = get_cell_is_dirichlet(get_fe_space(f))

FESpaces.num_dirichlet_dofs(f::ExtensionFESpace) = num_dirichlet_dofs(get_fe_space(f))

FESpaces.num_dirichlet_tags(f::ExtensionFESpace) = num_dirichlet_tags(get_fe_space(f))

FESpaces.get_dirichlet_dof_tag(f::ExtensionFESpace) = get_dirichlet_dof_tag(get_fe_space(f))

FESpaces.get_vector_type(f::ExtensionFESpace) = get_vector_type(get_fe_space(f))

function FESpaces.scatter_free_and_dirichlet_values(f::ExtensionFESpace,fv,dv)
  scatter_free_and_dirichlet_values(get_fe_space(f),fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::ExtensionFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,get_fe_space(f),cv)
end

function FESpaces.FEFunction(f::ExtensionFESpace,fv::AbstractVector,dv::AbstractVector)
  FEFunction(get_fe_space(f),fv,dv)
end

function get_dof_map(f::ExtensionFESpace,args...)
  bg_dof_to_act_dof = get_bg_dof_to_act_dof(f,args...)
  bg_ndofs = length(bg_dof_to_act_dof)
  return VectorDofMap(bg_ndofs,bg_dof_to_act_dof)
end

function _get_bg_odof_to_act_odof(bg_f::CartesianFESpace,act_f::CartesianFESpace)
  bg_bg_dof_to_bg_dof = get_bg_dof_to_act_dof(bg_f) # potential underlying constraints
  bg_dof_to_bg_bg_dof = findall(!iszero,bg_bg_dof_to_bg_dof)
  bg_bg_dof_to_act_dof = zeros(Int,length(bg_bg_dof_to_bg_dof))
  bg_cell_ids = get_cell_dof_ids(bg_f)
  act_cell_ids = get_cell_dof_ids(act_f)
  bg_cache = array_cache(bg_cell_ids)
  act_cache = array_cache(act_cell_ids)
  act_to_bg_cell = _get_bg_cell_to_act_cell(act_f)
  for (act_cell,bg_cell) in enumerate(act_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    act_dofs = getindex!(act_cache,act_cell_ids,act_cell)
    for (bg_dof,act_dof) in zip(bg_dofs,act_dofs)
      if bg_dof > 0
        bg_bg_dof = bg_dof_to_bg_bg_dof[bg_dof]
        bg_bg_dof_to_act_dof[bg_bg_dof] = act_dof
      end
    end
  end
  return bg_bg_dof_to_act_dof
end
