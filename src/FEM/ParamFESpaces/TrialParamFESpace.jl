"""
    TrialParamFESpace{S} <: SingleFieldParamFESpace{S}

Most standard implementation of a parametric trial FE space

"""
struct TrialParamFESpace{S} <: SingleFieldParamFESpace{S}
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
  TrialParamFESpace(ConsecutiveParamArray(dirichlet_values),space)
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
  dirichlet_values = consecutive_param_array(dv,N)
  TrialParamFESpace(dirichlet_values,U)
end

function HomogeneousTrialParamFESpace!(dirichlet_values::AbstractParamVector,U::SingleFieldFESpace,args...)
  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  TrialParamFESpace(dirichlet_values,U)
end

FESpaces.get_fe_space(f::TrialParamFESpace) = f.space

FESpaces.get_dirichlet_dof_values(f::TrialParamFESpace) = f.dirichlet_values

FESpaces.ConstraintStyle(::Type{<:TrialParamFESpace{U}}) where U = ConstraintStyle(U)

ParamDataStructures.param_length(f::TrialParamFESpace) = param_length(f.dirichlet_values)

function ParamDataStructures.param_getindex(f::TrialParamFESpace,index::Integer)
  dv = param_getindex(f.dirichlet_values,index)
  TrialFESpace(dv,f.space)
end

function FESpaces.FEFunction(
  tf::TrialParamFESpace{<:ZeroMeanFESpace},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector)

  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  c = FESpaces._compute_new_fixedval(
    free_values,
    dirichlet_values,
    f.vol_i,
    f.vol,
    f.space.dof_to_fix
  )
  fv = free_values + c
  dv = dirichlet_values + c
  FEFunction(tf′,fv,dv)
end

function FESpaces.EvaluationFunction(tf::TrialParamFESpace{<:ZeroMeanFESpace},free_values)
  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  FEFunction(tf′,free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(
  tf::TrialParamFESpace{<:FESpaceWithLinearConstraints},
  fmdof_to_val,
  dmdof_to_val)

  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  fdof_to_val = zero_free_values(tf′)
  ddof_to_val = zero_dirichlet_values(tf′)
  FESpaces._setup_dof_to_val!(
    fdof_to_val,
    ddof_to_val,
    fmdof_to_val,
    dmdof_to_val,
    f.DOF_to_mDOFs,
    f.DOF_to_coeffs,
    f.n_fdofs,
    f.n_fmdofs)
  scatter_free_and_dirichlet_values(tf′,fdof_to_val,ddof_to_val)
end

function FESpaces.gather_free_and_dirichlet_values(
  tf::TrialParamFESpace{<:FESpaceWithConstantFixed{T}},
  cv) where T<:FESpaces.FixConstant

  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  _fv,_dv = gather_free_and_dirichlet_values(tf′,cv)
  @assert innerlength(_dv) == 0
  fv = ParamVectorWithEntryRemoved(_fv,f.dof_to_fix)
  dv = get_param_entry(_fv,f.dof_to_fix:f.dof_to_fix)
  (fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fv,
  dv,
  tf::TrialParamFESpace{<:FESpaceWithConstantFixed{T}},
  cv) where T<:FESpaces.FixConstant

  @assert innerlength(dv) == 1
  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  _dv = similar(dv,eltype(dv),0)
  _fv = ParamVectorWithEntryRemoved(fv,f.dof_to_fix,zero(eltype(fv)))
  gather_free_and_dirichlet_values!(_fv,_dv,tf′,cv)
  dv[1] = _fv.value
  (fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fmdof_to_val,
  dmdof_to_val,
  tf::TrialParamFESpace{<:FESpaceWithLinearConstraints},
  cell_to_ludof_to_val)

  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  fdof_to_val,ddof_to_val = gather_free_and_dirichlet_values(tf′,cell_to_ludof_to_val)
  FESpaces._setup_mdof_to_val!(
    fmdof_to_val,
    dmdof_to_val,
    fdof_to_val,
    ddof_to_val,
    f.mDOF_to_DOF,
    f.n_fdofs,
    f.n_fmdofs)
  fmdof_to_val,dmdof_to_val
end

function FESpaces.FEFunction(
  tf::TrialParamFESpace{<:TProductFESpace},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector)

  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  FEFunction(tf′,free_values,dirichlet_values)
end

function FESpaces.EvaluationFunction(tf::TrialParamFESpace{<:TProductFESpace},free_values)
  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  EvaluationFunction(tf′,free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(tf::TrialParamFESpace{<:TProductFESpace},fv,dv)
  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  scatter_free_and_dirichlet_values(tf′,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(tf::TrialParamFESpace{<:TProductFESpace},cv)
  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  gather_free_and_dirichlet_values(tf′,cv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fv,
  dv,
  tf::TrialParamFESpace{<:TProductFESpace},
  cv)

  f = tf.space
  tf′ = TrialParamFESpace(f.space,tf.dirichlet_values)
  gather_free_and_dirichlet_values!(fv,dv,tf′,cv)
end
