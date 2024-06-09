function length_dirichlet_values end
length_dirichlet_values(f::FESpace) = @abstractmethod

function length_free_values end
length_free_values(f::FESpace) = length_dirichlet_values(f)

function get_dirichlet_cells end
get_dirichlet_cells(f::FESpace) = @abstractmethod
get_dirichlet_cells(f::UnconstrainedFESpace) = f.dirichlet_cells

abstract type SingleFieldParamFESpace <: SingleFieldFESpace end

FESpaces.get_free_dof_ids(f::SingleFieldParamFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::SingleFieldParamFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::SingleFieldParamFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::SingleFieldParamFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::SingleFieldParamFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::SingleFieldParamFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::SingleFieldParamFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::SingleFieldParamFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::SingleFieldParamFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::SingleFieldParamFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::SingleFieldParamFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::SingleFieldParamFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::SingleFieldParamFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::SingleFieldParamFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::SingleFieldParamFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

get_dirichlet_cells(f::SingleFieldParamFESpace) = get_dirichlet_cells(f.space)

# These functions allow us to use global ParamArrays

function FESpaces.get_vector_type(f::SingleFieldParamFESpace)
  V = get_vector_type(f.space)
  N = length_free_values(f)
  typeof(array_of_similar_arrays(V(),N))
end

function FESpaces.gather_free_and_dirichlet_values!(
  free_vals,
  dirichlet_vals,
  f::SingleFieldParamFESpace,
  cell_vals)

  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  cells = 1:length(cell_vals)

  FESpaces._free_and_dirichlet_values_fill!(
    free_vals,
    dirichlet_vals,
    cache_vals,
    cache_dofs,
    cell_vals,
    cell_dofs,
    cells)

  (free_vals,dirichlet_vals)
end

function FESpaces.gather_dirichlet_values!(
  dirichlet_vals,
  f::SingleFieldParamFESpace,
  cell_vals)

  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  free_vals = zero_free_values(f)
  cells = get_dirichlet_cells(f)

  FESpaces._free_and_dirichlet_values_fill!(
    free_vals,
    dirichlet_vals,
    cache_vals,
    cache_dofs,
    cell_vals,
    cell_dofs,
    cells)

  dirichlet_vals
end

function FESpaces._fill_dirichlet_values_for_tag!(
  dirichlet_values::AbstractParamVector,
  dv::AbstractParamVector,
  tag,
  dirichlet_dof_to_tag)

  @check param_length(dirichlet_values) == param_length(dv)
  for dof in 1:_innerlength(dv)
    if dirichlet_dof_to_tag[dof] == tag
      dirichlet_values.data[dof,:] .= dv.data[dof,:]
    end
  end
end

function FESpaces._free_and_dirichlet_values_fill!(
  free_vals::AbstractParamVector,
  dirichlet_vals::AbstractParamVector,
  cache_vals,
  cache_dofs,
  cell_vals,
  cell_dofs,
  cells)

  @check param_length(free_vals) == param_length(dirichlet_vals)
  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    for (i,dof) in enumerate(dofs)
      val = vals.data[i,:]
      if dof > 0
        free_vals.data[dof,:] .= val
      elseif dof < 0
        dirichlet_vals.data[-dof,:] .= val
      else
        @unreachable "dof ids either positive or negative, not zero"
      end
    end
  end
end

# This artifact aims to make a FESpace behave like a ParamFESpace with free and
# dirichlet values being ParamArrays of length L

struct FESpaceToParamFESpace{S,L} <: SingleFieldParamFESpace
  space::S
  FESpaceToParamFESpace(space::S,::Val{L}) where {S,L} = new{S,L}(space)
end

FESpaceToParamFESpace(f::SingleFieldFESpace,L::Integer) = FESpaceToParamFESpace(f,Val{L}())
FESpaceToParamFESpace(f::SingleFieldParamFESpace,L::Integer) = f

FESpaces.ConstraintStyle(::Type{<:FESpaceToParamFESpace{S}}) where S = ConstraintStyle(S)

length_dirichlet_values(f::FESpaceToParamFESpace{S,L}) where {S,L} = L

# Extend some of Gridap's functions when needed
function FESpaces.FEFunction(
  f::FESpaceToParamFESpace{<:ZeroMeanFESpace},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector)

  zf = f.space
  fv = similar(free_values)
  dv = similar(dirichlet_values)
  @check param_length(fv) == param_length(dv)
  @inbounds for i = param_eachindex(fv)
    fv_i = param_view(fv,i)
    dv_i = param_view(dv,i)
    c = FESpaces._compute_new_fixedval(
      fv_i,dv_i,zf.vol_i,zf.vol,zf.space.dof_to_fix)
    fv_i .+= c
    dv_i .+= c
  end
  f′ = FESpaceToParamFESpace(zf.space,length_dirichlet_values(f))
  FEFunction(f′,fv,dv)
end

function FESpaces.EvaluationFunction(f::FESpaceToParamFESpace{<:ZeroMeanFESpace},free_values)
  zf = f.space
  f′ = FESpaceToParamFESpace(zf.space,length_dirichlet_values(f))
  FEFunction(f′,free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:TrialFESpace},
  fv::AbstractParamVector,
  dv::AbstractParamVector)

  tf = f.space
  f′ = FESpaceToParamFESpace(tf.space,length_dirichlet_values(f))
  scatter_free_and_dirichlet_values(f′,fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:ZeroMeanFESpace},
  fv::AbstractParamVector,
  dv::AbstractParamVector)

  zf = f.space
  f′ = FESpaceToParamFESpace(zf.space,length_dirichlet_values(f))
  scatter_free_and_dirichlet_values(f′,fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:FESpaceWithConstantFixed{T}},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector
  ) where T<:FESpaces.FixConstant

  ff = f.space
  fv = similar(free_values)
  dv = similar(dirichlet_values)
  @check param_length(fv) == param_length(dv)
  @inbounds for i = param_eachindex(fv)
    fv_i = param_view(fv,i)
    dv_i = param_view(dv,i)
    fv_i .= FESpaces.VectorWithEntryInserted(fv_i,ff.dof_to_fix,dv_i[1])
    dv_i .= similar(dv_i,eltype(dv_i),0)
  end
  f′ = FESpaceToParamFESpace(ff.space,length_dirichlet_values(f))
  FEFunction(f′,fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:FESpaceWithConstantFixed{T}},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector
  ) where T<:FESpaces.DoNotFixConstant

  ff = f.space
  @check all(length.(dirichlet_values) .== 0)
  scatter_free_and_dirichlet_values(ff.space,free_values,dirichlet_values)
end

# for testing purposes

function FESpaces.test_single_field_fe_space(f::SingleFieldParamFESpace,pred=(==))
  fe_basis = get_fe_basis(f)
  @test isa(fe_basis,CellField)
  test_fe_space(f)
  dirichlet_values = zero_dirichlet_values(f)
  @test length(dirichlet_values) == num_dirichlet_dofs(f)
  free_values = zero_free_values(f)
  cell_vals = scatter_free_and_dirichlet_values(f,free_values,dirichlet_values)
  fv, dv = gather_free_and_dirichlet_values(f,cell_vals)
  @test pred(fv,free_values)
  @test pred(dv,dirichlet_values)
  gather_free_and_dirichlet_values!(fv,dv,f,cell_vals)
  @test pred(fv,free_values)
  @test pred(dv,dirichlet_values)
  fv, dv = gather_free_and_dirichlet_values!(fv,dv,f,cell_vals)
  @test pred(fv,free_values)
  @test pred(dv,dirichlet_values)
  fe_function = FEFunction(f,free_values,dirichlet_values)
  @test isa(fe_function,SingleFieldParamFEFunction)
  test_fe_function(fe_function)
  ddof_to_tag = get_dirichlet_dof_tag(f)
  @test length(ddof_to_tag) == num_dirichlet_dofs(f)
  if length(get_dirichlet_dof_tag(f)) != 0
    @test maximum(get_dirichlet_dof_tag(f)) <= num_dirichlet_tags(f)
  end
  cell_dof_basis = get_fe_dof_basis(f)
  @test isa(cell_dof_basis,CellDof)
end

function ParamDataStructures.param_getindex(f::FESpaceToParamFESpace,index::Integer)
  f.space
end
