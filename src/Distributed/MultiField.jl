struct DistributedMultiFieldPTFEFunction{A,B,C} <: GridapType
  field_fe_fun::A
  part_fe_fun::B
  free_values::C
  function DistributedMultiFieldPTFEFunction(
    field_fe_fun::AbstractVector{<:DistributedSingleFieldFEFunction},
    part_fe_fun::AbstractArray{<:MultiFieldPTFEFunction},
    free_values::AbstractVector)
    A = typeof(field_fe_fun)
    B = typeof(part_fe_fun)
    C = typeof(free_values)
    new{A,B,C}(field_fe_fun,part_fe_fun,free_values)
  end
end

function FESpaces.get_free_dof_values(uh::DistributedMultiFieldPTFEFunction)
  uh.free_values
end

GridapDistributed.local_views(a::DistributedMultiFieldPTFEFunction) = a.part_fe_fun
MultiField.num_fields(m::DistributedMultiFieldPTFEFunction) = length(m.field_fe_fun)
Base.iterate(m::DistributedMultiFieldPTFEFunction) = iterate(m.field_fe_fun)
Base.iterate(m::DistributedMultiFieldPTFEFunction,state) = iterate(m.field_fe_fun,state)
Base.getindex(m::DistributedMultiFieldPTFEFunction,field_id::Integer) = m.field_fe_fun[field_id]

struct DistributedMultiFieldPFESpace{A,B,C,D} <: DistributedFESpace
  field_fe_space::A
  part_fe_space::B
  gids::C
  vector_type::Type{D}
  function DistributedMultiFieldPFESpace(
    field_fe_space::AbstractVector{<:DistributedSingleFieldFESpace},
    part_fe_space::AbstractArray{<:MultiFieldPFESpace},
    gids::PRange,
    vector_type::Type{D}) where D
    A = typeof(field_fe_space)
    B = typeof(part_fe_space)
    C = typeof(gids)
    new{A,B,C,D}(field_fe_space,part_fe_space,gids,vector_type)
  end
end

GridapDistributed.local_views(a::DistributedMultiFieldPFESpace) = a.part_fe_space
MultiField.num_fields(m::DistributedMultiFieldPFESpace) = length(m.field_fe_space)
Base.iterate(m::DistributedMultiFieldPFESpace) = iterate(m.field_fe_space)
Base.iterate(m::DistributedMultiFieldPFESpace,state) = iterate(m.field_fe_space,state)
Base.getindex(m::DistributedMultiFieldPFESpace,field_id::Integer) = m.field_fe_space[field_id]
Base.length(m::DistributedMultiFieldPFESpace) = length(m.field_fe_space)

function FESpaces.get_vector_type(fs::DistributedMultiFieldPFESpace)
  fs.vector_type
end

function FESpaces.get_free_dof_ids(fs::DistributedMultiFieldPFESpace)
  fs.gids
end

function MultiField.restrict_to_field(
  f::DistributedMultiFieldPFESpace,free_values::PVector,field::Integer)
  values = map(f.part_fe_space,partition(free_values)) do u,x
    restrict_to_field(u,x,field)
  end
  gids = f.field_fe_space[field].gids
  PVector(values,partition(gids))
end

function FESpaces.FEFunction(
  f::DistributedMultiFieldPFESpace,x::AbstractVector,isconsistent=false)
  free_values = change_ghost(x,f.gids)
  local_vals = consistent_local_views(free_values,f.gids,isconsistent)
  part_fe_fun = map(FEFunction,f.part_fe_space,local_vals)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i,true)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldPTFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.EvaluationFunction(
  f::DistributedMultiFieldPFESpace,x::AbstractVector,isconsistent=false)
  free_values = change_ghost(x,f.gids)
  local_vals = consistent_local_views(free_values,f.gids,false)
  part_fe_fun = map(EvaluationFunction,f.part_fe_space,local_vals)
  field_fe_fun = DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = restrict_to_field(f,free_values,i)
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = EvaluationFunction(fe_space_i,free_values_i)
    push!(field_fe_fun,fe_fun_i)
  end
  DistributedMultiFieldPTFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate(objects,fe::DistributedMultiFieldPFESpace)
  free_values = zero_free_values(fe)
  interpolate!(objects,free_values,fe)
end

function FESpaces.interpolate!(objects,free_values::AbstractVector,fe::DistributedMultiFieldPFESpace)
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
  DistributedMultiFieldPTFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere(objects,fe::DistributedMultiFieldPFESpace)
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
  DistributedMultiFieldPTFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere!(
  objects,free_values::AbstractVector,
  dirichlet_values::Vector{AbstractArray{<:AbstractVector}},
  fe::DistributedMultiFieldPFESpace)
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
  DistributedMultiFieldPTFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.interpolate_everywhere(
  objects::Vector{<:DistributedCellDatum},fe::DistributedMultiFieldPFESpace)
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
  DistributedMultiFieldPTFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function FESpaces.get_fe_basis(f::DistributedMultiFieldPFESpace)
  part_mbasis = map(get_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = map(b->b[i],part_mbasis)
    bi = DistributedCellField(basis_i)
    push!(field_fe_basis,bi)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_mbasis)
end

function FESpaces.get_trial_fe_basis(f::DistributedMultiFieldPFESpace)
  part_mbasis = map(get_trial_fe_basis,f.part_fe_space)
  field_fe_basis = DistributedCellField[]
  for i in 1:num_fields(f)
    basis_i = map(b->b[i],part_mbasis)
    bi = DistributedCellField(basis_i)
    push!(field_fe_basis,bi)
  end
  DistributedMultiFieldFEBasis(field_fe_basis,part_mbasis)
end

function FEM.TrialPFESpace(objects,a::DistributedMultiFieldPFESpace)
  TransientTrialPFESpace(a,objects)
end

function FEM.TrialPFESpace(a::DistributedMultiFieldPFESpace,objects)
  f_dspace_test = a.field_fe_space
  f_dspace = map( arg -> TrialFESpace(arg[1],arg[2]), zip(f_dspace_test,objects) )
  f_p_space = map(local_views,f_dspace)
  v(x...) = collect(x)
  p_f_space = map(v,f_p_space...)
  p_mspace = map(MultiFieldPFESpace,p_f_space)
  gids = a.gids
  vector_type = a.vector_type
  DistributedMultiFieldPFESpace(f_dspace,p_mspace,gids,vector_type)
end
