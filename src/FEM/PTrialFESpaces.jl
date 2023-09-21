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
  dv = zero_dirichlet_values(U)
  array = Vector{typeof(dv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(dv)
  end
  dirichlet_values = PTArray(array)
  PTrialFESpace(dirichlet_values,U)
end

function HomogeneousPTrialFESpace(U::PTrialFESpace)
  dirichlet_values = zero_dirichlet_values(U)
  PTrialFESpace(dirichlet_values,U)
end

function HomogeneousPTrialFESpace!(
  dirichlet_values::PTArray,
  U::SingleFieldFESpace)

  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  PTrialFESpace(dirichlet_values,U)
end

FESpaces.get_free_dof_ids(f::PTrialFESpace) = get_free_dof_ids(f.space)

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

FESpaces.num_dirichlet_tags(f::PTrialFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::PTrialFESpace) = get_dirichlet_dof_tag(f.space)

function FESpaces.get_dirichlet_dof_values(f::PTrialFESpace)
  f.dirichlet_values
end

function FESpaces.zero_free_values(f::PTrialFESpace)
  fv = zero_free_values(f.space)
  n = length(f.dirichlet_values)
  array = Vector{typeof(fv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(fv)
  end
  PTArray(array)
end

function FESpaces.zero_dirichlet_values(f::PTrialFESpace)
  zdv = zero_dirichlet_values(f.space)
  n = length(f.dirichlet_values)
  array = Vector{typeof(zdv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(zdv)
  end
  PTArray(array)
end

for fe in (:PTrialFESpace,:TrialFESpace,:DirichletFESpace,:ZeroMeanFESpace)
  @eval begin
    function FESpaces.scatter_free_and_dirichlet_values(f::$fe,fv::PTArray,dv::PTArray)
      map((x,y)->scatter_free_and_dirichlet_values(f.space,x,y),fv,dv)
    end
  end
end

for fe in (:ConstantFESpace,:UnconstrainedFESpace)
  @eval begin
    function FESpaces.scatter_free_and_dirichlet_values(f::$fe,fv::PTArray,dv::PTArray)
      map((x,y)->scatter_free_and_dirichlet_values(f,x,y),fv,dv)
    end
  end
end

function FESpaces.gather_free_and_dirichlet_values!(
  free_vals::PTArray,
  dirichlet_vals::PTArray,
  f::SingleFieldFESpace,
  cell_vals::PTArray)

  cell_dofs = get_cell_dof_ids(f)
  cache_dofs = array_cache(cell_dofs)
  cell_vals1 = testitem(cell_vals)
  cache_vals = array_cache(cell_vals1)
  cells = eachindex(cell_vals1)

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
  dirichlet_vals::PTArray,
  f::SingleFieldFESpace,
  cell_vals::PTArray)

  cell_dofs = get_cell_dof_ids(f)
  cache_dofs = array_cache(cell_dofs)
  cell_vals1 = testitem(cell_vals)
  cache_vals = array_cache(cell_vals1)
  free_vals = zero_free_values(f)
  cells = f.dirichlet_cells

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
  free_vals::PTArray,
  dirichlet_vals::PTArray,
  cache_vals,
  cache_dofs,
  cell_vals::PTArray,
  cell_dofs,
  cells)

  @inbounds for cell in cells
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    for n in eachindex(free_vals)
      vals = getindex!(cache_vals,cell_vals[n],cell)
      for (i,dof) in enumerate(dofs)
        val = vals[i]
        if dof > 0
          free_vals[n][dof] = val
        elseif dof < 0
          dirichlet_vals[n][-dof] = val
        else
          @unreachable "dof ids either positive or negative, not zero"
        end
      end
    end
  end
end

function FESpaces._free_and_dirichlet_values_fill!(
  free_vals::AffinePTArray,
  dirichlet_vals::AffinePTArray,
  cache_vals,
  cache_dofs,
  cell_vals::AffinePTArray,
  cell_dofs,
  cells)

  fv = free_vals.array
  dv = dirichlet_vals.array
  cv = cell_vals.array
  _free_and_dirichlet_values_fill!(fv,dv,cache_vals,cache_dofs,cv,cell_dofs,cells)
end

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values::PTArray{T},
  dirichlet_values_scratch::PTArray{T},
  f::PTrialFESpace,
  tag_to_object) where T

  dirichlet_dof_to_tag = get_dirichlet_dof_tag(f)
  @inbounds for n in eachindex(dirichlet_values)
    dv = dirichlet_values[n]
    dvs = dirichlet_values_scratch[n]
    _tag_to_object = FESpaces._convert_to_collectable(tag_to_object[n],num_dirichlet_tags(f))
    fill!(dvs,zero(eltype(T)))
    for (tag, object) in enumerate(_tag_to_object)
      cell_vals = FESpaces._cell_vals(f,object)
      gather_dirichlet_values!(dvs,f.space,cell_vals)
      FESpaces._fill_dirichlet_values_for_tag!(dv,dvs,tag,dirichlet_dof_to_tag)
    end
  end
  test_ptarray(dirichlet_values)
  dirichlet_values
end

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values::AffinePTArray,
  dirichlet_values_scratch::AffinePTArray,
  f::PTrialFESpace,
  tag_to_object)

  dv = dirichlet_values.array
  dvs = dirichlet_values_scratch.array
  @assert get_affinity(tag_to_object) == Affine()
  tto = first(tag_to_object)
  _free_and_dirichlet_values_fill!(dv,dvs,f,tto)
end

Arrays.testitem(f::FESpace) = f
Arrays.testitem(f::PTrialFESpace) = TrialFESpace(testitem(f.dirichlet_values),f.space)
Arrays.testitem(f::MultiFieldFESpace) = MultiFieldFESpace(map(testitem,f.spaces))
