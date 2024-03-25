struct TrialParamFESpace{S} <: SingleFieldParamFESpace
  dirichlet_values::ParamArray
  space::S
  function TrialParamFESpace(dirichlet_values::ParamArray,space::SingleFieldFESpace)
    new{typeof(space)}(dirichlet_values,space)
  end
end

function TrialParamFESpace(U::SingleFieldFESpace)
  U
end

function TrialParamFESpace(dirichlet_values::AbstractVector,space::SingleFieldFESpace)
  TrialFESpace(dirichlet_values,space)
end

function TrialParamFESpace(space::SingleFieldFESpace,objects)
  dirichlet_values = [compute_dirichlet_values_for_tags(space,objects[i]) for i = 1:length(objects)]
  TrialParamFESpace(ParamArray(dirichlet_values),space)
end

function TrialParamFESpace!(dir_values::ParamArray,space::SingleFieldFESpace,objects)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  TrialParamFESpace!(dir_values,space)
end

function TrialParamFESpace!(space::TrialParamFESpace,objects)
  dir_values = get_dirichlet_dof_values(space)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  space
end

# Allow do-block syntax

function TrialParamFESpace(f::Function,space::SingleFieldFESpace)
  TrialParamFESpace(space,f)
end

function TrialParamFESpace!(f::Function,dir_values::ParamArray,space::SingleFieldFESpace)
  TrialParamFESpace!(dir_values,space,f)
end

function TrialParamFESpace!(f::Function,space::TrialParamFESpace)
  TrialParamFESpace!(space,f)
end

function HomogeneousTrialParamFESpace(U::SingleFieldFESpace,::Val{N}) where N
  dv = zero_dirichlet_values(U)
  dirichlet_values = allocate_param_array(dv,N)
  TrialParamFESpace(dirichlet_values,U)
end

function HomogeneousTrialParamFESpace!(dirichlet_values::ParamArray,U::SingleFieldFESpace,args...)
  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  TrialParamFESpace(dirichlet_values,U)
end

# Delegated functions

FESpaces.get_dirichlet_dof_values(f::TrialParamFESpace) = f.dirichlet_values

FESpaces.ConstraintStyle(::Type{<:TrialParamFESpace{U}}) where U = ConstraintStyle(U)

length_dirichlet_values(f::TrialParamFESpace) = length(f.dirichlet_values)

function _getindex(f::TrialParamFESpace,index)
  dv = f.dirichlet_values[index]
  TrialFESpace(dv,f.space)
end
