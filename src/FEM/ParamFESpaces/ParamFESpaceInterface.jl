function get_dirichlet_cells end
get_dirichlet_cells(f::FESpace) = @abstractmethod
get_dirichlet_cells(f::UnconstrainedFESpace) = f.dirichlet_cells
get_dirichlet_cells(f::TProductFESpace) = get_dirichlet_cells(f.space)

ParamDataStructures.param_length(f::FESpace) = 0

"""
    abstract type SingleFieldParamFESpace <: SingleFieldFESpace end

Parametric extension of a [`SingleFieldFESpace`](@ref) in [`Gridap`](@ref). The
FE spaces inhereting are (trial) spaces on which we can easily define a
[`ParamFEFunction`](@ref). Most commonly, a SingleFieldParamFESpace is
characterized by parametric Dirichlet boundary conditions, but a standard
nonparametric DBC can be prescribed on such spaces.


Subtypes:
- TrivialParamFESpace{S,L} <: SingleFieldParamFESpace
- TrialParamFESpace{S} <: SingleFieldParamFESpace

"""
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

function get_vector_index_map(f::SingleFieldParamFESpace)
  get_vector_index_map(f.space)
end

function get_matrix_index_map(f::SingleFieldParamFESpace,g::SingleFieldFESpace)
  get_matrix_index_map(f.space,g)
end

get_dirichlet_cells(f::SingleFieldParamFESpace) = get_dirichlet_cells(f.space)

# These functions allow us to use global ParamArrays

function FESpaces.get_vector_type(f::SingleFieldParamFESpace)
  V = get_vector_type(f.space)
  L = param_length(f)
  typeof(array_of_consecutive_arrays(V(),L))
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

  println(typeof(dirichlet_values))
  println(typeof(dv))
  @check param_length(dirichlet_values) == param_length(dv)
  for dof in 1:_innerlength(dv)
    if dirichlet_dof_to_tag[dof] == tag
      @inbounds for k in param_eachindex(dirichlet_values)
        dirichlet_values.data[k][dof] = dv.data[k][dof]
      end
    end
  end
end

function FESpaces._fill_dirichlet_values_for_tag!(
  dirichlet_values::ConsecutiveVectorOfVectors,
  dv::ConsecutiveVectorOfVectors,
  tag,
  dirichlet_dof_to_tag)

  @check param_length(dirichlet_values) == param_length(dv)
  for dof in 1:_innerlength(dv)
    if dirichlet_dof_to_tag[dof] == tag
      @inbounds for k in param_eachindex(dirichlet_values)
        dirichlet_values.data[dof,k] = dv.data[dof,k]
      end
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
      @inbounds for k in param_eachindex(dirichlet_vals)
        val = vals.data[k][i]
        if dof > 0
          free_vals.data[k][dof] = val
        elseif dof < 0
          dirichlet_vals.data[k][-dof] = val
        else
          @unreachable "dof ids either positive or negative, not zero"
        end
      end
    end
  end
end

function FESpaces._free_and_dirichlet_values_fill!(
  free_vals::ConsecutiveVectorOfVectors,
  dirichlet_vals::ConsecutiveVectorOfVectors,
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
      @inbounds for k in param_eachindex(dirichlet_vals)
        val = vals.data[k][i]
        if dof > 0
          free_vals.data[dof,k] = val
        elseif dof < 0
          dirichlet_vals.data[-dof,k] = val
        else
          @unreachable "dof ids either positive or negative, not zero"
        end
      end
    end
  end
end

"""
    TrivialParamFESpace{S,L} <: SingleFieldParamFESpace

Wrapper for nonparametric FE spaces that we wish assumed a parametric length `L`

"""

struct TrivialParamFESpace{S,L} <: SingleFieldParamFESpace
  space::S
  TrivialParamFESpace(space::S,::Val{L}) where {S,L} = new{S,L}(space)
end

TrivialParamFESpace(space::FESpace,plength::Integer) = TrivialParamFESpace(space,Val{plength}())

ParamDataStructures.param_length(f::TrivialParamFESpace{S,L}) where {S,L} = L
ParamDataStructures.to_param_quantity(f::SingleFieldParamFESpace,plength::Integer) = f
ParamDataStructures.to_param_quantity(f::SingleFieldFESpace,plength::Integer) = TrivialParamFESpace(f,Val{plength}())
ParamDataStructures.param_getindex(f::TrivialParamFESpace,index::Integer) = f.space

FESpaces.ConstraintStyle(::Type{<:TrivialParamFESpace{S}}) where S = ConstraintStyle(S)

# Extend some of Gridap's functions when needed
function FESpaces.FEFunction(
  f::TrivialParamFESpace{<:ZeroMeanFESpace},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector)

  zf = f.space
  @check param_length(free_values) == param_length(dirichlet_values)
  fv,dv = map(param_data(free_values),param_data(dirichlet_values)) do fv,dv
    c = FESpaces._compute_new_fixedval(fv,dv,zf.vol_i,zf.vol,zf.space.dof_to_fix)
    _fv = lazy_map(+,fv,Fill(c,length(fv)))
    _dv = dv .+ c
    return _fv,_dv
  end |> tuple_of_arrays
  f′ = TrivialParamFESpace(zf.space,param_length(f))
  FEFunction(f′,ParamArray(fv),ParamArray(dv))
end

function FESpaces.EvaluationFunction(f::TrivialParamFESpace{<:ZeroMeanFESpace},free_values)
  zf = f.space
  f′ = TrivialParamFESpace(zf.space,param_length(f))
  FEFunction(f′,free_values)
end

function FESpaces.EvaluationFunction(f::TrivialParamFESpace{<:TrialFESpace{<:ZeroMeanFESpace}},free_values)
  zf = f.space
  f′ = TrivialParamFESpace(zf.space,param_length(f))
  FEFunction(f′,free_values)
end

function FESpaces.EvaluationFunction(f::TrivialParamFESpace{<:TrivialParamFESpace{<:ZeroMeanFESpace}},free_values)
  zf = f.space
  f′ = TrivialParamFESpace(zf.space,param_length(f))
  FEFunction(f′,free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::TrivialParamFESpace{<:TrialFESpace},
  fv::AbstractParamVector,
  dv::AbstractParamVector)

  tf = f.space
  f′ = TrivialParamFESpace(tf.space,param_length(f))
  scatter_free_and_dirichlet_values(f′,fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::TrivialParamFESpace{<:TProductFESpace},
  fv::AbstractParamVector,
  dv::AbstractParamVector)

  tf = f.space
  f′ = TrivialParamFESpace(tf.space,param_length(f))
  scatter_free_and_dirichlet_values(f′,fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::TrivialParamFESpace{<:ZeroMeanFESpace},
  fv::AbstractParamVector,
  dv::AbstractParamVector)

  zf = f.space
  f′ = TrivialParamFESpace(zf.space,param_length(f))
  scatter_free_and_dirichlet_values(f′,fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::TrivialParamFESpace{<:FESpaceWithConstantFixed{T}},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector
  ) where T<:FESpaces.FixConstant

  ff = f.space
  @check param_length(free_values) == param_length(dirichlet_values)
  fv,dv = map(param_data(free_values),param_data(dirichlet_values)) do fv,dv
    _fv = FESpaces.VectorWithEntryInserted(fv,ff.dof_to_fix,dv[1])
    _dv = similar(dv,eltype(dv),0)
    return _fv,_dv
  end |> tuple_of_arrays
  f′ = TrivialParamFESpace(ff.space,param_length(f))
  scatter_free_and_dirichlet_values(f′,ParamArray(fv),ParamArray(dv))
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::TrivialParamFESpace{<:FESpaceWithConstantFixed{T}},
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
