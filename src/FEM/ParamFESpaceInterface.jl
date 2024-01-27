function length_dirichlet_values end
length_dirichlet_values(f::FESpace) = @abstractmethod

function length_free_values end
length_free_values(f::FESpace) = length_dirichlet_values(f)

function get_dirichlet_cells end
get_dirichlet_cells(f::FESpace) = @abstractmethod
get_dirichlet_cells(f::UnconstrainedFESpace) = f.dirichlet_cells

abstract type SingleFieldParamFESpace{S} <: SingleFieldFESpace end

FESpaces.get_free_dof_ids(f::SingleFieldParamFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::SingleFieldParamFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::SingleFieldParamFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::SingleFieldParamFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::SingleFieldParamFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::SingleFieldParamFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::SingleFieldParamFESpace) = get_fe_dof_basis(f.space)

FESpaces.ConstraintStyle(::Type{<:SingleFieldParamFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_cell_isconstrained(f::SingleFieldParamFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::SingleFieldParamFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::SingleFieldParamFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::SingleFieldParamFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::SingleFieldParamFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::SingleFieldParamFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::SingleFieldParamFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::SingleFieldParamFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

get_dirichlet_cells(f::SingleFieldParamFESpace) = get_dirichlet_cells(f.space)

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values,
  dirichlet_values_scratch,
  f::SingleFieldParamFESpace,
  tag_to_object)

  dirichlet_dof_to_tag = get_dirichlet_dof_tag(f)
  _tag_to_object = FESpaces._convert_to_collectable(tag_to_object,num_dirichlet_tags(f))
  map(dirichlet_values,dirichlet_values_scratch,_tag_to_object) do dv,dvs,tto
    fill!(dvs,zero(eltype(dirichlet_values_scratch)))
    for (tag,object) in enumerate(tto)
      cell_vals = FESpaces._cell_vals(f,object)
      gather_dirichlet_values!(dvs,f.space,cell_vals)
      FESpaces._fill_dirichlet_values_for_tag!(dv,dvs,tag,dirichlet_dof_to_tag)
    end
  end
  ParamArray(dirichlet_values)
end

function FESpaces._convert_to_collectable(object::AbstractParamFunction,ntags)
  objects = map(object) do o
    FESpaces._convert_to_collectable(Fill(o,ntags),ntags)
  end
  ParamArray(objects)
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

# Extend some of Gridap's functions when needed

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceWithConstantFixed{FESpaces.FixConstant},
  fv::ParamArray,
  dv::ParamArray)

  _dv,_fv = map(fv,dv) do fv,dv
    @assert length(dv) == 1
    _dv = similar(dv,eltype(dv),0)
    _fv = FESpaces.VectorWithEntryInserted(fv,f.dof_to_fix,dv[1])
    _dv,_fv
  end |> tuple_of_arrays
  scatter_free_and_dirichlet_values(f.space,ParamArray(_fv),ParamArray(_dv))
end

function FESpaces.gather_free_and_dirichlet_values(
  f::FESpaceWithConstantFixed{FESpaces.FixConstant},
  cv::LazyArray{<:Any,<:ParamArray})

  _fv,_dv = gather_free_and_dirichlet_values(f.space,cv)
  fv,dv = map(_fv,_dv) do _fv,_dv
    @assert length(_dv) == 0
    fv = FESpaces.VectorWithEntryRemoved(_fv,f.dof_to_fix)
    dv = _fv[f.dof_to_fix:f.dof_to_fix]
    fv,dv
  end |> tuple_of_arrays
  ParamArray(fv),ParamArray(dv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fv::ParamArray,
  dv::ParamArray,
  f::FESpaceWithConstantFixed{FESpaces.FixConstant},
  cv::LazyArray{<:Any,<:ParamArray})

  _fv,_dv = gather_free_and_dirichlet_values(f.space,cv)
  fv,dv = map(fv,dv,_fv,_dv) do fv,dv,_fv,_dv
    @assert length(_dv) == 0
    fv    .= FESpaces.VectorWithEntryRemoved(_fv,f.dof_to_fix)
    dv[1]  = _fv[f.dof_to_fix]
  end |> tuple_of_arrays
  ParamArray(fv),ParamArray(dv)
end

# This artifact aims to make a FESpace behave like a ParamFESpace with free and
# dirichlet values being ParamArrays of length L

struct FESpaceToParamFESpace{S,L} <: SingleFieldParamFESpace{S}
  space::S
  FESpaceToParamFESpace(space::S,::Val{L}) where {S,L} = new{S,L}(space)
end

FESpaceToParamFESpace(f::SingleFieldFESpace,L::Integer) = FESpaceToParamFESpace(f,Val(L))
FESpaceToParamFESpace(f::SingleFieldParamFESpace,L::Integer) = f

length_dirichlet_values(f::FESpaceToParamFESpace{S,L}) where {S,L} = L

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
