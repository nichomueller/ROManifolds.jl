# Interface for time marching across parameters

struct PTTrialFESpace{S} <: SingleFieldFESpace
  dirichlet_values::PTArray
  space::S
  function PTTrialFESpace(dirichlet_values::PTArray,space::SingleFieldFESpace)
    new{typeof(space)}(dirichlet_values,space)
  end
end

function PTTrialFESpace(space::SingleFieldFESpace,objects)
  dirichlet_values = compute_dirichlet_values_for_tags(space,objects)
  PTTrialFESpace(dirichlet_values,space)
end

function PTTrialFESpace!(dir_values::PTArray,space::SingleFieldFESpace,objects)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  PTTrialFESpace(dir_values,space)
end

function PTTrialFESpace!(space::PTTrialFESpace,objects)
  dir_values = get_dirichlet_dof_values(space)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  space
end

function HomogeneousPTTrialFESpace(U::SingleFieldFESpace,n::Int)
  dirichlet_values = PTArray(zero_dirichlet_values(U),n)
  PTTrialFESpace(dirichlet_values,U)
end

function HomogeneousPTTrialFESpace(U::PTTrialFESpace)
  dirichlet_values = zero_dirichlet_values(U)
  PTTrialFESpace(dirichlet_values,U)
end

function HomogeneousPTTrialFESpace!(dirichlet_values::PTArray,U::SingleFieldFESpace)
  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  PTTrialFESpace(dirichlet_values,U)
end

FESpaces.get_dirichlet_dof_values(f::PTTrialFESpace) = f.dirichlet_values

FESpaces.get_free_dof_ids(f::PTTrialFESpace) = get_free_dof_ids(f.space)

FESpaces.zero_free_values(f::PTTrialFESpace) = zero_free_values(f.space)

FESpaces.get_triangulation(f::PTTrialFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::PTTrialFESpace) = get_dof_value_type(f.space)

FESpaces.get_vector_type(f::PTTrialFESpace) = get_vector_type(f.space)

FESpaces.get_cell_dof_ids(f::PTTrialFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::PTTrialFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::PTTrialFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::PTTrialFESpace) = get_fe_dof_basis(f.space)

FESpaces.ConstraintStyle(::Type{<:PTTrialFESpace{B}}) where B = ConstraintStyle(B)

FESpaces.get_cell_isconstrained(f::PTTrialFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::PTTrialFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::PTTrialFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::PTTrialFESpace) = get_cell_is_dirichlet(f.space)

function FESpaces.zero_dirichlet_values(f::PTTrialFESpace)
  n = length(f.dirichlet_values)
  uD0 = zero_dirichlet_values(f.space)
  PTArray(uD0,n)
end

FESpaces.num_dirichlet_tags(f::PTTrialFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::PTTrialFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::PTTrialFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

FESpaces.gather_free_and_dirichlet_values(f::PTTrialFESpace,cv) = gather_free_and_dirichlet_values(f.space,cv)

FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::PTTrialFESpace,cv) = gather_free_and_dirichlet_values!(fv,dv,f.space,cv)

FESpaces.gather_dirichlet_values(f::PTTrialFESpace,cv) = gather_dirichlet_values(f.space,cv)

FESpaces.gather_dirichlet_values!(dv,f::PTTrialFESpace,cv) = gather_dirichlet_values!(dv,f.space,cv)

FESpaces.gather_free_values(f::PTTrialFESpace,cv) = gather_free_values(f.space,cv)

FESpaces.gather_free_values!(fv,f::PTTrialFESpace,cv) = gather_free_values!(fv,f.space,cv)

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values::PTArray,
  dirichlet_values_scratch::PTArray,
  f::SingleFieldFESpace,
  tag_to_object::PTArray)

  dv = zero(dirichlet_values).array
  dvs = zero(dirichlet_values_scratch).array
  tto = tag_to_object.array

  dirichlet_dof_to_tag = get_dirichlet_dof_tag(f)
  k = 0
  for (_dv,_dvs,_tto) in zip(dv,dvs,tto)
    k += 1
    _tag_to_object = FESpaces._convert_to_collectable(_tto,num_dirichlet_tags(f))
    for (tag,object) in enumerate(_tag_to_object)
      cell_vals = FESpaces._cell_vals(f,object)
      fill!(_dvs,zero(eltype(_dvs)))
      gather_dirichlet_values!(_dvs,f,cell_vals)
      FESpaces._fill_dirichlet_values_for_tag!(_dv,_dvs,tag,dirichlet_dof_to_tag)
    end
    dirichlet_values.array[k] = _dv
  end
  dirichlet_values
end
