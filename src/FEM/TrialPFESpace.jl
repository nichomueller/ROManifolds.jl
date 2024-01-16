struct TrialPFESpace{S} <: SingleFieldFESpace
  dirichlet_values::PArray
  space::S
  function TrialPFESpace(dirichlet_values::PArray,space::SingleFieldFESpace)
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
  dirichlet_values = compute_dirichlet_values_for_tags(space,objects)
  TrialPFESpace(dirichlet_values,space)
end

function TrialPFESpace!(dir_values::PArray,space::SingleFieldFESpace,objects)
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

function TrialPFESpace!(f::Function,dir_values::PArray,space::SingleFieldFESpace)
  TrialPFESpace!(dir_values,space,f)
end

function TrialPFESpace!(f::Function,space::TrialPFESpace)
  TrialPFESpace!(space,f)
end

function HomogeneousTrialPFESpace(U::SingleFieldFESpace,n::Int)
  dv = zero_dirichlet_values(U)
  dirichlet_values = parray(dv,n)
  TrialPFESpace(dirichlet_values,U)
end

function HomogeneousTrialPFESpace!(dirichlet_values::PArray,U::SingleFieldFESpace,args...)
  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  TrialPFESpace(dirichlet_values,U)
end

FESpaces.get_free_dof_ids(f::TrialPFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::TrialPFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::TrialPFESpace) = get_dof_value_type(f.space)

FESpaces.get_vector_type(f::TrialPFESpace) = get_vector_type(f.space)

FESpaces.get_cell_dof_ids(f::TrialPFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::TrialPFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::TrialPFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::TrialPFESpace) = get_fe_dof_basis(f.space)

FESpaces.ConstraintStyle(::Type{<:TrialPFESpace{B}}) where B = ConstraintStyle(B)

FESpaces.get_cell_isconstrained(f::TrialPFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::TrialPFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::TrialPFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::TrialPFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_tags(f::TrialPFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::TrialPFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_dirichlet_dof_values(f::TrialPFESpace) = f.dirichlet_values

FESpaces.scatter_free_and_dirichlet_values(f::TrialPFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

# These functions allow us to pass from cell-wise PArray(s) to global PArray(s)
function FESpaces.zero_free_values(f::TrialPFESpace)
  fv = zero_free_values(f.space)
  n = length(f.dirichlet_values)
  array = Vector{typeof(fv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(fv)
  end
  PArray(array)
end

function FESpaces.zero_dirichlet_values(f::TrialPFESpace)
  zdv = zero_dirichlet_values(f.space)
  n = length(f.dirichlet_values)
  array = Vector{typeof(zdv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(zdv)
  end
  PArray(array)
end

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values::PArray{T},
  dirichlet_values_scratch::PArray{T},
  f::TrialPFESpace,
  tag_to_object) where T

  dirichlet_dof_to_tag = get_dirichlet_dof_tag(f)
  @inbounds for n in eachindex(dirichlet_values)
    dv = dirichlet_values[n]
    dvs = dirichlet_values_scratch[n]
    _tag_to_object = FESpaces._convert_to_collectable(tag_to_object[n],num_dirichlet_tags(f))
    fill!(dvs,zero(eltype(T)))
    for (tag,object) in enumerate(_tag_to_object)
      cell_vals = FESpaces._cell_vals(f,object)
      gather_dirichlet_values!(dvs,f.space,cell_vals)
      FESpaces._fill_dirichlet_values_for_tag!(dv,dvs,tag,dirichlet_dof_to_tag)
    end
  end
  dirichlet_values
end

function FESpaces.gather_free_and_dirichlet_values!(
  free_vals,
  dirichlet_vals,
  f::TrialPFESpace,
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
  f::TrialPFESpace,
  cell_vals)

  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
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
  free_vals::PArray,
  dirichlet_vals::PArray,
  cache_vals,
  cache_dofs,
  cell_vals::PArray,
  cell_dofs::PArray,
  cells)

  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    for (i,dof) in enumerate(dofs)
      for k in eachindex(vals)
        val = vals[k][i]
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

function FESpaces._cell_vals(fs::TrialPFESpace,object)
  s = get_fe_dof_basis(fs)
  trian = get_triangulation(s)
  f = CellField(object,trian,DomainStyle(s))
  cell_vals = s(f)
end

# function FESpaces.interpolate!(
#   object::AbstractPFunction,
#   free_values::PArray,
#   fs::TrialPFESpace)

#   for k in eachindex(object)
#     cell_vals = FESpaces._cell_vals(fs,object[k])
#     gather_free_values!(free_values[k],fs,cell_vals)
#   end
#   FEFunction(fs,free_values)
# end

# function FESpaces.interpolate_everywhere!(
#   object::AbstractPFunction,
#   free_values::PArray,
#   dirichlet_values::PArray,
#   fs::TrialPFESpace)

#   for k in eachindex(object)
#     cell_vals = FESpaces._cell_vals(fs,object[k])
#     gather_free_and_dirichlet_values!(free_values[k],dirichlet_values[k],fs,cell_vals)
#   end
#   FEFunction(fs,free_values,dirichlet_values)
# end

# function FESpaces.interpolate_dirichlet!(
#   object::AbstractPFunction,
#   free_values::PArray,
#   dirichlet_values::PArray,
#   fs::TrialPFESpace)

#   for k in eachindex(object)
#     cell_vals = FESpaces._cell_vals(fs,object[k])
#     gather_dirichlet_values!(dirichlet_values[k],fs,cell_vals)
#     fill!(free_values[k],zero(eltype(free_values[k])))
#   end
#   FEFunction(fs,free_values,dirichlet_values)
# end
