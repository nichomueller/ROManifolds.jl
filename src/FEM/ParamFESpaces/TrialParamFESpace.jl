"""
    TrialParamFESpace{S} <: SingleFieldParamFESpace

Most standard implementation of a parametric trial FE space

"""
struct TrialParamFESpace{S} <: SingleFieldParamFESpace
  dirichlet_values::AbstractParamVector
  space::S
  function TrialParamFESpace(dirichlet_values::AbstractParamVector,space::SingleFieldFESpace)
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
  dirichlet_values = [compute_dirichlet_values_for_tags(space,objects[i]) for i = param_eachindex(objects)]
  TrialParamFESpace(ConsecutiveArrayOfArrays(dirichlet_values),space)
end

function TrialParamFESpace!(dir_values::AbstractParamVector,space::SingleFieldFESpace,objects)
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

function TrialParamFESpace!(f::Function,dir_values::AbstractParamVector,space::SingleFieldFESpace)
  TrialParamFESpace!(dir_values,space,f)
end

function TrialParamFESpace!(f::Function,space::TrialParamFESpace)
  TrialParamFESpace!(space,f)
end

function HomogeneousTrialParamFESpace(U::SingleFieldFESpace,::Val{N}) where N
  dv = zero_dirichlet_values(U)
  dirichlet_values = array_of_consecutive_arrays(dv,N)
  TrialParamFESpace(dirichlet_values,U)
end

function HomogeneousTrialParamFESpace!(dirichlet_values::AbstractParamVector,U::SingleFieldFESpace,args...)
  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  TrialParamFESpace(dirichlet_values,U)
end

FESpaces.get_dirichlet_dof_values(f::TrialParamFESpace) = f.dirichlet_values

FESpaces.ConstraintStyle(::Type{<:TrialParamFESpace{U}}) where U = ConstraintStyle(U)

ParamDataStructures.param_length(f::TrialParamFESpace) = param_length(f.dirichlet_values)

function ParamDataStructures.param_getindex(f::TrialParamFESpace,index::Integer)
  dv = param_getindex(f.dirichlet_values,index)
  TrialFESpace(dv,f.space)
end
