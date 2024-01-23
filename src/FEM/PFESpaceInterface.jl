function length_dirichlet_values end
length_dirichlet_values(f::FESpace) = @abstractmethod

function length_free_values end
length_free_values(f::FESpace) = length_dirichlet_values(f)

function get_dirichlet_cells end
get_dirichlet_cells(f::FESpace) = @abstractmethod
get_dirichlet_cells(f::UnconstrainedFESpace) = f.dirichlet_cells

abstract type SingleFieldPFESpace{S} <: SingleFieldFESpace end

FESpaces.get_free_dof_ids(f::SingleFieldPFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::SingleFieldPFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::SingleFieldPFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::SingleFieldPFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::SingleFieldPFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::SingleFieldPFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::SingleFieldPFESpace) = get_fe_dof_basis(f.space)

FESpaces.ConstraintStyle(::Type{<:SingleFieldPFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_cell_isconstrained(f::SingleFieldPFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::SingleFieldPFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::SingleFieldPFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::SingleFieldPFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::SingleFieldPFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::SingleFieldPFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::SingleFieldPFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::SingleFieldPFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

get_dirichlet_cells(f::SingleFieldPFESpace) = get_dirichlet_cells(f.space)

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values,
  dirichlet_values_scratch,
  f::SingleFieldPFESpace,
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
  f::SingleFieldPFESpace,
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
  f::SingleFieldPFESpace,
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

# These functions allow us to use global PArray(s)

function FESpaces.get_vector_type(f::SingleFieldPFESpace)
  V = get_vector_type(f.space)
  N = length_free_values(f)
  typeof(PArray{V}(undef,N))
end

# function FESpaces.zero_free_values(f::SingleFieldPFESpace)
#   V = get_vector_type(f)
#   allocate_vector(V,num_free_dofs(f))
# end

# function FESpaces.zero_dirichlet_values(f::SingleFieldPFESpace)
#   V = get_vector_type(f)
#   allocate_vector(V,num_dirichlet_dofs(f))
# end

# This artifact aims to make a FESpace behave like a PFESpace with free and
# dirichlet values being PArrays of length L

struct FESpaceToPFESpace{S,L} <: SingleFieldPFESpace{S}
  space::S
  FESpaceToPFESpace(space::S,::Val{L}) where {S,L} = new{S,L}(space)
end

FESpaceToPFESpace(f::SingleFieldFESpace,L::Integer) = FESpaceToPFESpace(f,Val(L))
FESpaceToPFESpace(f::SingleFieldPFESpace,L::Integer) = f

length_dirichlet_values(f::FESpaceToPFESpace{S,L}) where {S,L} = L

# for visualization/testing purposes

function FESpaces.test_single_field_fe_space(f::SingleFieldPFESpace,pred=(==))
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
  @test isa(fe_function,SingleFieldFEPFunction)
  test_fe_function(fe_function)
  ddof_to_tag = get_dirichlet_dof_tag(f)
  @test length(ddof_to_tag) == num_dirichlet_dofs(f)
  if length(get_dirichlet_dof_tag(f)) != 0
    @test maximum(get_dirichlet_dof_tag(f)) <= num_dirichlet_tags(f)
  end
  cell_dof_basis = get_fe_dof_basis(f)
  @test isa(cell_dof_basis,CellDof)
end

function _getindex(f::FESpaceToPFESpace,index)
  f.space
end
