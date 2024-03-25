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
  dirichlet_values::ParamArray,
  dv::ParamArray,
  tag,
  dirichlet_dof_to_tag)

  @assert length(dirichlet_values) == length(dv)
  for i = eachindex(dirichlet_values)
    FESpaces._fill_dirichlet_values_for_tag!(dirichlet_values[i],dv[i],tag,dirichlet_dof_to_tag)
  end
end

function FESpaces._free_and_dirichlet_values_fill!(
  free_vals::ParamArray,
  dirichlet_vals::ParamArray,
  cache_vals,
  cache_dofs,
  cell_vals,
  cell_dofs,
  cells)

  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    map(vals,free_vals,dirichlet_vals) do vals,free_vals,dirichlet_vals
      for (i,dof) in enumerate(dofs)
        val = vals[i]
        if dof > 0
          free_vals[dof] = val
        elseif dof < 0
          dirichlet_vals[-dof] = val
        else
          @unreachable "dof ids either positive or negative, not zero"
        end
      end
    end
  end

end

function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,free_values::ParamArray,dirichlet_values::ParamArray)
  cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
  cell_field = CellField(fs,cell_vals)
  SingleFieldParamFEFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)
end

# This function allows us to use global ParamArrays

function FESpaces.get_vector_type(f::SingleFieldParamFESpace)
  V = get_vector_type(f.space)
  N = length_free_values(f)
  typeof(ParamVector{V}(undef,N))
end

# This artifact aims to make a FESpace behave like a ParamFESpace with free and
# dirichlet values being ParamArrays of length L

struct FESpaceToParamFESpace{S,L} <: SingleFieldParamFESpace
  space::S
  FESpaceToParamFESpace(space::S,::Val{L}) where {S,L} = new{S,L}(space)
end

FESpaceToParamFESpace(f::SingleFieldFESpace,L::Integer) = FESpaceToParamFESpace(f,Val(L))
FESpaceToParamFESpace(f::SingleFieldParamFESpace,L::Integer) = f

FESpaces.ConstraintStyle(::Type{<:FESpaceToParamFESpace{S}}) where S = ConstraintStyle(S)

length_dirichlet_values(f::FESpaceToParamFESpace{S,L}) where {S,L} = L

# Extend some of Gridap's functions when needed
function FESpaces.FEFunction(
  f::FESpaceToParamFESpace{<:ZeroMeanFESpace,L},
  free_values::ParamArray,
  dirichlet_values::ParamArray) where L

  fs = f.space
  fv,dv = map(free_values,dirichlet_values) do _fv,_dv
    c = FESpaces._compute_new_fixedval(
      _fv,_dv,fs.vol_i,fs.vol,fs.space.dof_to_fix)
    fv = lazy_map(+,_fv,Fill(c,length(_fv)))
    dv = _dv .+ c
    fv,dv
  end |> tuple_of_arrays
  FEFunction(FESpaceToParamFESpace(fs.space,Val(L)),ParamArray(fv),ParamArray(dv))
end

function FESpaces.EvaluationFunction(
  f::FESpaceToParamFESpace{<:ZeroMeanFESpace,L},
  free_values) where L
  FEFunction(FESpaceToParamFESpace(f.space.space,Val(L)),free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:TrialFESpace,L},
  fv::ParamArray,
  dv::ParamArray) where L
  scatter_free_and_dirichlet_values(FESpaceToParamFESpace(f.space.space,Val(L)),fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:ZeroMeanFESpace,L},
  fv::ParamArray,
  dv::ParamArray) where L
  scatter_free_and_dirichlet_values(FESpaceToParamFESpace(f.space.space,Val(L)),fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:FESpaceWithConstantFixed{<:FESpaces.FixConstant}},
  fv::ParamArray,
  dv::ParamArray)

  fs = f.space
  _dv,_fv = map(fv,dv) do fv,dv
    @assert length(dv) == 1
    _dv = similar(dv,eltype(dv),0)
    _fv = FESpaces.VectorWithEntryInserted(fv,fs.dof_to_fix,dv[1])
    _dv,_fv
  end |> tuple_of_arrays
  scatter_free_and_dirichlet_values(fs.space,ParamArray(_fv),ParamArray(_dv))
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:FESpaceWithConstantFixed{<:FESpaces.DoNotFixConstant}},
  fv::ParamArray,
  dv::ParamArray)

  fs = f.space
  @assert all(length.(dv) .== 0)
  scatter_free_and_dirichlet_values(fs.space,ParamArray(fv),ParamArray(dv))
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

function _getindex(f::FESpaceToParamFESpace,index)
  f.space
end
