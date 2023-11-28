struct DistributedPTMultiFieldFEFunction{A,B,C} <: GridapType
  field_fe_fun::A
  part_fe_fun::B
  free_values::C
  function DistributedPTMultiFieldFEFunction(
    field_fe_fun::AbstractVector{<:DistributedSingleFieldFEFunction},
    part_fe_fun::AbstractArray{<:MultiFieldFEFunction},
    free_values::AbstractVector)
    A = typeof(field_fe_fun)
    B = typeof(part_fe_fun)
    C = typeof(free_values)
    new{A,B,C}(field_fe_fun,part_fe_fun,free_values)
  end
end


function FESpaces.get_free_dof_values(uh::DistributedPTMultiFieldFEFunction)
  uh.free_values
end

local_views(a::DistributedPTMultiFieldFEFunction) = a.part_fe_fun
MultiField.num_fields(m::DistributedPTMultiFieldFEFunction) = length(m.field_fe_fun)
Base.iterate(m::DistributedPTMultiFieldFEFunction) = iterate(m.field_fe_fun)
Base.iterate(m::DistributedPTMultiFieldFEFunction,state) = iterate(m.field_fe_fun,state)
Base.getindex(m::DistributedPTMultiFieldFEFunction,field_id::Integer) = m.field_fe_fun[field_id]

local_views(a::Vector{<:DistributedCellField}) = [ai.fields for ai in a]

struct DistributedPTMultiFieldFESpace{A,B,C,D} <: DistributedFESpace
  field_fe_space::A
  part_fe_space::B
  gids::C
  vector_type::Type{D}
  function DistributedPTMultiFieldFESpace(
    field_fe_space::AbstractVector{<:DistributedSingleFieldFESpace},
    part_fe_space::AbstractArray{<:MultiFieldFESpace},
    gids::PRange,
    vector_type::Type{D}) where D
    A = typeof(field_fe_space)
    B = typeof(part_fe_space)
    C = typeof(gids)
    new{A,B,C,D}(field_fe_space,part_fe_space,gids,vector_type)
  end
end

local_views(a::DistributedPTMultiFieldFESpace) = a.part_fe_space
MultiField.num_fields(m::DistributedPTMultiFieldFESpace) = length(m.field_fe_space)
Base.iterate(m::DistributedPTMultiFieldFESpace) = iterate(m.field_fe_space)
Base.iterate(m::DistributedPTMultiFieldFESpace,state) = iterate(m.field_fe_space,state)
Base.getindex(m::DistributedPTMultiFieldFESpace,field_id::Integer) = m.field_fe_space[field_id]
Base.length(m::DistributedPTMultiFieldFESpace) = length(m.field_fe_space)

function FESpaces.get_vector_type(fs::DistributedPTMultiFieldFESpace)
  fs.vector_type
end

function FESpaces.get_free_dof_ids(fs::DistributedPTMultiFieldFESpace)
  fs.gids
end

function MultiField.restrict_to_field(
  f::DistributedPTMultiFieldFESpace,free_values::PVector,field::Integer)
  values = map(f.part_fe_space,partition(free_values)) do u,x
    restrict_to_field(u,x,field)
  end
  gids = f.field_fe_space[field].gids
  PVector(values,partition(gids))
end

function FESpaces.FEFunction(
  f::DistributedPTMultiFieldFESpace,x::AbstractVector,isconsistent=false)
  free_values = change_ghost(x,f.gids)
  # This will cause also the single-field components to be consistent
  local_vals = consistent_local_views(free_values,f.gids,isconsistent)
  part_fe_fun = map(FEFunction,f.part_fe_space,local_vals)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i,true)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.EvaluationFunction(
  f::DistributedPTMultiFieldFESpace,x::AbstractVector,isconsistent=false)
  free_values = change_ghost(x,f.gids)
  # This will cause also the single-field components to be consistent
  local_vals = consistent_local_views(free_values,f.gids,false)
  part_fe_fun = map(EvaluationFunction,f.part_fe_space,local_vals)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = EvaluationFunction(fe_space_i,free_values_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate(objects,fe::DistributedPTMultiFieldFESpace)
  free_values = zero_free_values(fe)
  interpolate!(objects,free_values,fe)
end

function FESpaces.interpolate!(objects,free_values::AbstractVector,fe::DistributedPTMultiFieldFESpace)
  local_vals = consistent_local_views(free_values,fe.gids,true)
  part_fe_fun = map(local_vals,local_views(fe)) do x,f
    interpolate!(objects,x,f)
  end
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_space_i = fe.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere(objects,fe::DistributedPTMultiFieldFESpace)
  free_values = zero_free_values(fe)
  local_vals = consistent_local_views(free_values,fe.gids,true)
  part_fe_fun = map(local_vals,local_views(fe)) do x,f
    interpolate!(objects,x,f)
  end
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_space_i = fe.field_fe_space[i]
    dirichlet_values_i = zero_dirichlet_values(fe_space_i)
    fe_fun_i = interpolate_everywhere!(objects[i], free_values_i,dirichlet_values_i,fe_space_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere!(
  objects,free_values::AbstractVector,
  dirichlet_values::Vector{AbstractArray{<:AbstractVector}},
  fe::DistributedPTMultiFieldFESpace)
  local_vals = consistent_local_views(free_values,fe.gids,true)
  part_fe_fun = map(local_vals,local_views(fe)) do x,f
    interpolate!(objects,x,f)
  end
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = restrict_to_field(fe,free_values,i)
    dirichlet_values_i = dirichlet_values[i]
    fe_space_i = fe.field_fe_space[i]
    fe_fun_i = interpolate_everywhere!(objects[i], free_values_i,dirichlet_values_i,fe_space_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere(
  objects::Vector{<:DistributedCellDatum},fe::DistributedPTMultiFieldFESpace)
  local_objects = local_views(objects)
  local_spaces = local_views(fe)
  part_fe_fun = map(local_spaces,local_objects...) do f,o...
    interpolate_everywhere(o,f)
  end
  free_values = zero_free_values(fe)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_space_i = fe.field_fe_space[i]
    dirichlet_values_i = get_dirichlet_dof_values(fe_space_i)
    fe_fun_i = interpolate_everywhere!(objects[i], free_values_i,dirichlet_values_i,fe_space_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.get_fe_basis(f::DistributedPTMultiFieldFESpace)
  part_mbasis = map(get_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = map(b->b[i],part_mbasis)
    bi = DistributedCellField(basis_i)
    push!(field_fe_basis,bi)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_mbasis)
end

function FESpaces.get_trial_fe_basis(f::DistributedPTMultiFieldFESpace)
  part_mbasis = map(get_trial_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = map(b->b[i],part_mbasis)
    bi = DistributedCellField(basis_i)
    push!(field_fe_basis,bi)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_mbasis)
end

function PTrialFESpace(objects,a::DistributedPTMultiFieldFESpace)
  PTTrialFESpace(a,objects)
end

function PTrialFESpace(a::DistributedPTMultiFieldFESpace,objects)
  f_dspace_test = a.field_fe_space
  f_dspace = map( arg -> TrialFESpace(arg[1],arg[2]), zip(f_dspace_test,objects) )
  f_p_space = map(local_views,f_dspace)
  v(x...) = collect(x)
  p_f_space = map(v,f_p_space...)
  p_mspace = map(MultiFieldFESpace,p_f_space)
  gids = a.gids
  vector_type = a.vector_type
  DistributedPTMultiFieldFESpace(f_dspace,p_mspace,gids,vector_type)
end
