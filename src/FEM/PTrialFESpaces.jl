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
  dirichlet_values = PTArray(zero_dirichlet_values(U),n)
  PTrialFESpace(dirichlet_values,U)
end

function HomogeneousPTrialFESpace(U::PTrialFESpace)
  dirichlet_values = zero_dirichlet_values(U)
  PTrialFESpace(dirichlet_values,U)
end

function HomogeneousPTrialFESpace!(
  dirichlet_values::PTArray{T},
  U::SingleFieldFESpace) where T

  fill!(dirichlet_values,zero(eltype(T)))
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
  n = length(f.dirichlet_values)
  PTArray(zero_free_values(f.space),n)
end

function FESpaces.zero_dirichlet_values(f::PTrialFESpace)
  zero(f.dirichlet_values)
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
  f::FESpace,
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
  f::FESpace,
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

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values::PTArray{T},
  dirichlet_values_scratch::PTArray{T},
  f::PTrialFESpace,
  tag_to_object) where T

  dv = zeros(dirichlet_values)
  dvs = zeros(dirichlet_values_scratch)
  dirichlet_dof_to_tag = get_dirichlet_dof_tag(f)
  for n in eachindex(dirichlet_values)
    _tag_to_object = FESpaces._convert_to_collectable(tag_to_object[n],num_dirichlet_tags(f))
    fill!(dvs[n],zero(eltype(T)))
    for (tag, object) in enumerate(_tag_to_object)
      cell_vals = FESpaces._cell_vals(f,object)
      gather_dirichlet_values!(dvs[n],f.space,cell_vals)
      FESpaces._fill_dirichlet_values_for_tag!(dv[n],dvs[n],tag,dirichlet_dof_to_tag)
    end
    dirichlet_values[n] = dv[n]
  end
  dirichlet_values
end
