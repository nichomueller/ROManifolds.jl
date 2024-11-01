"""
    TrivialParamFESpace{S} <: SingleFieldParamFESpace{S}

Wrapper for nonparametric FE spaces that we wish assumed a parametric length `plength`

"""
struct TrivialParamFESpace{S} <: SingleFieldParamFESpace{S}
  space::S
  plength::Int
end

FESpaces.get_fe_space(f::TrivialParamFESpace) = f.space

FESpaces.ConstraintStyle(::Type{<:TrivialParamFESpace{S}}) where S = ConstraintStyle(S)

ParamDataStructures.param_length(f::TrivialParamFESpace) = f.plength
ParamDataStructures.to_param_quantity(f::SingleFieldParamFESpace,plength::Integer) = f
ParamDataStructures.to_param_quantity(f::SingleFieldFESpace,plength::Integer) = TrivialParamFESpace(f,plength)
ParamDataStructures.param_getindex(f::TrivialParamFESpace,index::Integer) = f.space

function FESpaces.TrialFESpace(tf::TrivialParamFESpace)
  f = tf.space
  U = TrialFESpace(f)
  TrivialParamFESpace(U,param_length(tf))
end

function FESpaces.FEFunction(
  tf::TrivialParamFESpace{<:ZeroMeanFESpace},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector)

  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(tf))
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

function FESpaces.EvaluationFunction(tf::TrivialParamFESpace{<:ZeroMeanFESpace},free_values)
  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(tf))
  FEFunction(tf′,free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(
  tf::TrivialParamFESpace{<:FESpaceWithLinearConstraints},
  fmdof_to_val,
  dmdof_to_val)

  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(tf))
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
  tf::TrivialParamFESpace{<:FESpaceWithConstantFixed{T}},
  cv) where T<:FESpaces.FixConstant

  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(tf))
  _fv,_dv = gather_free_and_dirichlet_values(tf′,cv)
  @assert innerlength(_dv) == 0
  fv = ParamVectorWithEntryRemoved(_fv,f.dof_to_fix)
  dv = get_param_entry(_fv,f.dof_to_fix:f.dof_to_fix)
  (fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fv,
  dv,
  tf::TrivialParamFESpace{<:FESpaceWithConstantFixed{T}},
  cv) where T<:FESpaces.FixConstant

  @assert innerlength(dv) == 1
  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(tf))
  _dv = similar(dv,eltype(dv),0)
  _fv = ParamVectorWithEntryRemoved(fv,f.dof_to_fix,zero(eltype(fv)))
  gather_free_and_dirichlet_values!(_fv,_dv,tf′,cv)
  dv[1] = _fv.value
  (fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fmdof_to_val,
  dmdof_to_val,
  tf::TrivialParamFESpace{<:FESpaceWithLinearConstraints},
  cell_to_ludof_to_val)

  f = tf.space
  tf′ = TrivialParamFESpace(f.space,param_length(tf))
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
