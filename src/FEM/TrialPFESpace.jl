struct TrialPFESpace{S} <: SingleFieldPFESpace{S}
  dirichlet_values::ParamArray
  space::S
  function TrialPFESpace(dirichlet_values::ParamArray,space::SingleFieldFESpace)
    new{typeof(space)}(dirichlet_values,space)
  end
end

function TrialPFESpace(U::SingleFieldFESpace)
  U
end

function TrialPFESpace(dirichlet_values::AbstractVector,space::SingleFieldFESpace)
  @notimplemented
end

function TrialPFESpace(space::SingleFieldFESpace,objects)
  dirichlet_values = map(objects) do object
    compute_dirichlet_values_for_tags(space,object)
  end
  TrialPFESpace(ParamArray(dirichlet_values),space)
end

function TrialPFESpace!(dir_values::ParamArray,space::SingleFieldFESpace,objects)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  TrialPFESpace!(dir_values,space)
end

function TrialPFESpace!(space::TrialPFESpace,objects)
  dir_values = get_dirichlet_dof_values(space)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  space
end

# Allow do-block syntax

function TrialPFESpace(f::Function,space::SingleFieldFESpace)
  TrialPFESpace(space,f)
end

function TrialPFESpace!(f::Function,dir_values::ParamArray,space::SingleFieldFESpace)
  TrialPFESpace!(dir_values,space,f)
end

function TrialPFESpace!(f::Function,space::TrialPFESpace)
  TrialPFESpace!(space,f)
end

function HomogeneousTrialPFESpace(U::SingleFieldFESpace,::Val{N}) where N
  dv = zero_dirichlet_values(U)
  dirichlet_values = allocate_parray(dv,N)
  TrialPFESpace(dirichlet_values,U)
end

function HomogeneousTrialPFESpace!(dirichlet_values::ParamArray,U::SingleFieldFESpace,args...)
  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  TrialPFESpace(dirichlet_values,U)
end

# Delegated functions

FESpaces.get_dirichlet_dof_values(f::TrialPFESpace) = f.dirichlet_values

length_dirichlet_values(f::TrialPFESpace) = length(f.dirichlet_values)
