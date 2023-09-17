# Interface for time marching across parameters

struct PTrialFESpace{S} <: SingleFieldFESpace
  dirichlet_values::PTArray
  space::S
  function PTrialFESpace(dirichlet_values::PTArray,space::SingleFieldFESpace)
    new{typeof(space)}(dirichlet_values,space)
  end
end

function PTrialFESpace(space::SingleFieldFESpace,objects)
  dirichlet_values = compute_dirichlet_values_for_tags(space,objects)
  PTrialFESpace(dirichlet_values,space)
end

function PTrialFESpace!(dir_values::PTArray,space::SingleFieldFESpace,objects)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  PTrialFESpace!(dir_values,space)
end

function PTrialFESpace!(space::PTrialFESpace,objects)
  dir_values = get_dirichlet_dof_values(space)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  space
end

function HomogeneousPTrialFESpace(U::SingleFieldFESpace,n::Int)
  dirichlet_values = PTArray(zero_dirichlet_values(U),n)
  PTrialFESpace(dirichlet_values,U)
end

function HomogeneousPTrialFESpace(U::PTrialFESpace)
  dirichlet_values = zero_dirichlet_values(U)
  PTrialFESpace(dirichlet_values,U)
end

function HomogeneousPTrialFESpace!(
  dirichlet_values::PTArray{T},
  U::SingleFieldFESpace) where T

  fill!(dirichlet_values,zero(eltype(T)))
  PTrialFESpace(dirichlet_values,U)
end

FESpaces.get_dirichlet_dof_values(f::PTrialFESpace) = f.dirichlet_values

FESpaces.get_free_dof_ids(f::PTrialFESpace) = get_free_dof_ids(f.space)

FESpaces.zero_free_values(f::PTrialFESpace) = zero_free_values(f.space)

FESpaces.get_triangulation(f::PTrialFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::PTrialFESpace) = get_dof_value_type(f.space)

FESpaces.get_vector_type(f::PTrialFESpace) = get_vector_type(f.space)

FESpaces.get_cell_dof_ids(f::PTrialFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::PTrialFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::PTrialFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::PTrialFESpace) = get_fe_dof_basis(f.space)

FESpaces.ConstraintStyle(::Type{<:PTrialFESpace{B}}) where B = ConstraintStyle(B)

FESpaces.get_cell_isconstrained(f::PTrialFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::PTrialFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::PTrialFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::PTrialFESpace) = get_cell_is_dirichlet(f.space)

function FESpaces.zero_dirichlet_values(f::PTrialFESpace)
  n = length(f.dirichlet_values)
  uD0 = zero_dirichlet_values(f.space)
  PTArray(uD0,n)
end

FESpaces.num_dirichlet_tags(f::PTrialFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::PTrialFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::PTrialFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

FESpaces.gather_free_and_dirichlet_values(f::PTrialFESpace,cv) = gather_free_and_dirichlet_values(f.space,cv)

FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::PTrialFESpace,cv) = gather_free_and_dirichlet_values!(fv,dv,f.space,cv)

FESpaces.gather_dirichlet_values(f::PTrialFESpace,cv) = gather_dirichlet_values(f.space,cv)

FESpaces.gather_dirichlet_values!(dv,f::PTrialFESpace,cv) = gather_dirichlet_values!(dv,f.space,cv)

FESpaces.gather_free_values(f::PTrialFESpace,cv) = gather_free_values(f.space,cv)

FESpaces.gather_free_values!(fv,f::PTrialFESpace,cv) = gather_free_values!(fv,f.space,cv)

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values::PTArray{T},
  dirichlet_values_scratch::PTArray{T},
  f::SingleFieldFESpace,
  tag_to_object) where T

  dv = zeros(dirichlet_values)
  dvs = zeros(dirichlet_values_scratch)

  dirichlet_dof_to_tag = get_dirichlet_dof_tag(f)
  k = 0
  for (_dv,_dvs,_tto) in zip(dv,dvs,tag_to_object)
    k += 1
    _tag_to_object = FESpaces._convert_to_collectable(_tto,num_dirichlet_tags(f))
    for (tag,object) in enumerate(_tag_to_object)
      cell_vals = FESpaces._cell_vals(f,object)
      fill!(_dvs,zero(eltype(T)))
      gather_dirichlet_values!(_dvs,f,cell_vals)
      FESpaces._fill_dirichlet_values_for_tag!(_dv,_dvs,tag,dirichlet_dof_to_tag)
    end
    dirichlet_values[k] = _dv
  end
  dirichlet_values
end

function FESpaces.interpolate_everywhere(object::PFunction,f::SingleFieldFESpace)
  p = object.params
  np = length(p)
  free_values = PTArray(zero_free_values(f),np)
  dirichlet_values = PTArray(zero_dirichlet_values(f),np)
  fv = zeros(free_values)
  dv = zeros(dirichlet_values_scratch)
  k = 0
  for (_fv,_dv) in zip(fv,dv)
    k += 1
    objk = object.f(p[k])
    cell_vals = _cell_vals(fs,objk)
    gather_free_and_dirichlet_values!(_fv,_dv,f,cell_vals)
    free_values[k] = _fv
    dirichlet_values[k] = _dv
  end
  FEFunction(f,free_values,dirichlet_values)
end

function MultiField.restrict_to_field(
  f::MultiFieldFESpace,
  free_values::PTArray,
  field::Integer)

  MultiField._restrict_to_field(f,MultiFieldStyle(f),free_values,field)
end

function MultiField._restrict_to_field(
  f,::ConsecutiveMultiFieldStyle,
  free_values::PTArray,
  field)

  offsets = compute_field_offsets(f)
  U = f.spaces
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  PTArray(map(fv -> SubVector(fv,pini,pend),free_values))
end
