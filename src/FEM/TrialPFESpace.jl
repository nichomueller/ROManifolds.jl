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

function TrialPFESpace(dirichlet_values::AbstractVector{<:AbstractVector},space::SingleFieldFESpace)
  TrialPFESpace(PArray(dirichlet_values),space)
end

function TrialPFESpace(space::SingleFieldFESpace,objects::AbstractPFunction)
  dirichlet_values = map(objects) do object
    compute_dirichlet_values_for_tags(space,object)
  end
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

function HomogeneousTrialPFESpace(U::SingleFieldFESpace,::Val{N}) where N
  dv = zero_dirichlet_values(U)
  dirichlet_values = allocate_parray(dv,N)
  TrialPFESpace(dirichlet_values,U)
end

function HomogeneousTrialPFESpace!(dirichlet_values::PArray,U::SingleFieldFESpace,args...)
  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  TrialPFESpace(dirichlet_values,U)
end

function length_dirichlet_values end
length_dirichlet_values(f::FESpace) = @abstractmethod
length_dirichlet_values(f::TrialPFESpace) = length(f.dirichlet_values)

function length_free_values end
length_free_values(f::FESpace) = length_dirichlet_values(f)

FESpaces.get_free_dof_ids(f::TrialPFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::TrialPFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::TrialPFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::TrialPFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::TrialPFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::TrialPFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::TrialPFESpace) = get_fe_dof_basis(f.space)

FESpaces.ConstraintStyle(::Type{<:TrialPFESpace{B}}) where B = ConstraintStyle(B)

FESpaces.get_cell_isconstrained(f::TrialPFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::TrialPFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::TrialPFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::TrialPFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::TrialPFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::TrialPFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::TrialPFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_dirichlet_dof_values(f::TrialPFESpace) = f.dirichlet_values

FESpaces.scatter_free_and_dirichlet_values(f::TrialPFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

# These functions allow us to use global PArray(s)
function FESpaces.get_vector_type(f::TrialPFESpace)
  V = get_vector_type(f.space)
  N = length_free_values(f)
  typeof(PArray{V}(undef,N))
end

function FESpaces.zero_free_values(f::TrialPFESpace)
  V = get_vector_type(f)
  allocate_vector(V,num_dirichlet_dofs(f))
end

function FESpaces.zero_dirichlet_values(f::TrialPFESpace)
  V = get_vector_type(f)
  allocate_vector(V,num_dirichlet_dofs(f))
end

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values::PArray{T,N,A,L},
  dirichlet_values_scratch::PArray{T,N,A,L},
  f::TrialPFESpace,
  tag_to_object::AbstractPFunction) where {T,N,A,L}

  _tag_to_object = FESpaces._convert_to_collectable(tag_to_object,num_dirichlet_tags(f))
  dirichlet_dof_to_tag = get_dirichlet_dof_tag(f)
  map(dirichlet_values,dirichlet_values_scratch,_tag_to_object) do dv,dvs,tto
    fill!(dvs,zero(eltype(T)))
    for (tag,object) in enumerate(tto)
      cell_vals = FESpaces._cell_vals(f,object)
      gather_dirichlet_values!(dvs,f.space,cell_vals)
      FESpaces._fill_dirichlet_values_for_tag!(dv,dvs,tag,dirichlet_dof_to_tag)
    end
  end
  dirichlet_values
end

function FESpaces._convert_to_collectable(object::AbstractPFunction,ntags)
  objects = map(object) do o
    @assert typeof(o) <: Function
    FESpaces._convert_to_collectable(Fill(o,ntags),ntags)
  end
  PArray(objects)
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

# for visualization/testing purposes

function Base.iterate(f::TrialPFESpace)
  index = 1
  final_index = length(f.dirichlet_values)
  state = (index,final_index)
  dv = f.dirichlet_values[index]
  TrialFESpace(dv,f.space),state
end

function Base.iterate(f::TrialPFESpace,state)
  index,final_index = state
  index += 1
  if index > final_index
    return nothing
  end
  state = (index,final_index)
  dv = f.dirichlet_values[index]
  TrialFESpace(dv,f.space),state
end

function FESpaces.test_single_field_fe_space(f::TrialPFESpace,pred=(==))
  fe_basis = get_fe_basis(f)
  @test isa(fe_basis,CellField)
  test_fe_space(f)
  cell_dofs = get_cell_dof_ids(f)
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
  @test isa(fe_function,SingleFieldPFEFunction)
  test_fe_function(fe_function)
  ddof_to_tag = get_dirichlet_dof_tag(f)
  @test length(ddof_to_tag) == num_dirichlet_dofs(f)
  if length(get_dirichlet_dof_tag(f)) != 0
    @test maximum(get_dirichlet_dof_tag(f)) <= num_dirichlet_tags(f)
  end
  cell_dof_basis = get_fe_dof_basis(f)
  @test isa(cell_dof_basis,CellDof)
end
