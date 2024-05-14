function _minimum_dir_d(i::AbstractVector{CartesianIndex{D}},d::Integer) where D
  mind = Inf
  for ii in i
    if ii.I[d] < mind
      mind = ii.I[d]
    end
  end
  return mind
end

function _maximum_dir_d(i::AbstractVector{CartesianIndex{D}},d::Integer) where D
  maxd = 0
  for ii in i
    if ii.I[d] > maxd
      maxd = ii.I[d]
    end
  end
  return maxd
end

function _shape_per_dir(i::AbstractVector{CartesianIndex{D}}) where D
  function _admissible_shape(d::Int)
    mind = _minimum_dir_d(i,d)
    maxd = _maximum_dir_d(i,d)
    @assert all([ii.I[d] ≥ mind for ii in i]) && all([ii.I[d] ≤ maxd for ii in i])
    return maxd - mind + 1
  end
  ntuple(d -> _admissible_shape(d),D)
end

function _shape_per_dir(i::AbstractVector{<:Integer})
  min1 = minimum(i)
  max1 = maximum(i)
  (max1 - min1 + 1,)
end

function comp_to_free_dofs(::Type{T},space::FESpace,args...;kwargs...) where T
  @abstractmethod
end

function comp_to_free_dofs(::Type{T},space::UnconstrainedFESpace;kwargs...) where T
  glue = space.metadata
  ncomps = num_components(T)
  free_dof_to_comp = if isnothing(glue)
    _free_dof_to_comp(space,ncomps;kwargs...)
  else
    glue.free_dof_to_comp
  end
  comp_to_free_dofs(free_dof_to_comp,ncomps)
end

function _free_dof_to_comp(space,ncomps;kwargs...)
  @notimplemented
end

function comp_to_free_dofs(dof2comp,ncomps)
  comp2dof = Vector{typeof(dof2comp)}(undef,ncomps)
  for comp in 1:ncomps
    comp2dof[comp] = findall(dof2comp.==comp)
  end
  return comp2dof
end

function _get_cell_dof_comp_ids(cell_dof_ids,dofs)
  T = eltype(cell_dof_ids)
  ncells = length(cell_dof_ids)
  new_cell_ids = Vector{T}(undef,ncells)
  cache_cell_dof_ids = array_cache(cell_dof_ids)
  @inbounds for icell in 1:ncells
    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    ids_comp = findall(map(cd->cd ∈ dofs,abs.(cell_dofs)))
    new_cell_ids[icell] = cell_dofs[ids_comp]
  end
  return Table(new_cell_ids)
end

function _dof_perm_from_dof_perms(dof_perms::Vector{Matrix{Ti}}) where Ti
  @check all(size.(dof_perms) .== [size(first(dof_perms))])
  s = size(first(dof_perms))
  Dc = length(dof_perms)
  dof_perm = zeros(VectorValue{Dc,Ti},s)
  for ij in LinearIndices(s)
    perms_ij = getindex.(dof_perms,ij)
    dof_perm[ij] = Point(perms_ij)
  end
  return dof_perm
end

function _get_terms(p::Polytope,orders)
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  terms = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  return terms
end

function _get_dof_permutation(
  model::CartesianDiscreteModel{Dc},
  cell_dof_ids::Table,
  order::Integer) where Dc

  desc = get_cartesian_descriptor(model)

  periodic = desc.isperiodic
  ncells = desc.partition
  ndofs = order .* ncells .+ 1 .- periodic

  new_dof_ids = copy(LinearIndices(ndofs))

  terms = _get_terms(first(get_polytopes(model)),fill(order,Dc))
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  for (icell,cell) in enumerate(CartesianIndices(ncells))
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    new_dofs_range = map(i -> i:i+order,first_new_dof)
    new_dofs = view(new_dof_ids,new_dofs_range...)

    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    for (idof,dof) in enumerate(cell_dofs)
      t = terms[idof]
      new_dofs[t] < 0 && continue
      if dof < 0
        new_dofs[t] *= -1
      end
    end
  end

  pos_ids = findall(new_dof_ids.>0)
  neg_ids = findall(new_dof_ids.<0)
  new_dof_ids[pos_ids] .= LinearIndices(pos_ids)
  new_dof_ids[neg_ids] .= -1 .* LinearIndices(neg_ids)

  free_vals_shape = _shape_per_dir(pos_ids)
  n2o_dof_map = fill(-1,free_vals_shape)

  for (icell,cell) in enumerate(CartesianIndices(ncells))
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    new_dofs_range = map(i -> i:i+order,first_new_dof)
    new_dofs = view(new_dof_ids,new_dofs_range...)

    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    for (idof,dof) in enumerate(cell_dofs)
      t = terms[idof]
      new_dofs[t] < 0 && continue
      n2o_dof_map[new_dofs[t]] = dof
    end
  end

  return n2o_dof_map
end

function get_dof_permutation(
  ::Type{T},
  model::CartesianDiscreteModel,
  space::UnconstrainedFESpace,
  order::Integer;
  kwargs...) where T

  cell_dof_ids = get_cell_dof_ids(space)
  _get_dof_permutation(model,cell_dof_ids,order)
end

function get_dof_permutation(
  ::Type{T},
  model::CartesianDiscreteModel,
  space::UnconstrainedFESpace,
  order::Integer;
  kwargs...) where T<:MultiValue

  cell_dof_ids = get_cell_dof_ids(space)
  comp2dofs = comp_to_free_dofs(T,space;kwargs...)
  Ti = eltype(eltype(cell_dof_ids))
  dof_perms = Matrix{Ti}[]
  for dofs in comp2dofs
    cell_dof_comp_ids = _get_cell_dof_comp_ids(cell_dof_ids,dofs)
    dof_perm_comp = _get_dof_permutation(model,cell_dof_comp_ids,order)
    push!(dof_perms,dof_perm_comp)
  end
  dof_perm = _dof_perm_from_dof_perms(dof_perms)
  return dof_perm
end

function _get_tp_dof_permutation(models::AbstractVector,spaces::AbstractVector,order::Integer)
  @assert length(models) == length(spaces)
  D = length(models)
  function _tensor_product(aprev::AbstractArray{Tp,M},a::AbstractVector{Td}) where {Tp,Td,M}
    T = promote_type(Tp,Td)
    N = M+1
    s = (size(aprev)...,length(a))
    atp = zeros(T,s)
    slicesN = eachslice(atp,dims=N)
    @inbounds for (iN,sliceN) in enumerate(slicesN)
      sliceN .= aprev .+ a[iN]
    end
    return atp
  end
  function _d_dof_permutation(::Val{1},::Val{d′}) where d′
    @assert d′ == D
    model_d = models[1]
    space_d = spaces[1]
    ndofs_d = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
    ndofs = ndofs_d
    cell_ids_d = get_cell_dof_ids(space_d)
    dof_permutations_1d = _get_dof_permutation(model_d,cell_ids_d,order)
    return _d_dof_permutation(dof_permutations_1d,ndofs,Val(2),Val(d′-1))
  end
  function _d_dof_permutation(node2dof_prev,ndofs_prev,::Val{d},::Val{d′}) where {d,d′}
    model_d = models[d]
    space_d = spaces[d]
    ndofs_d = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
    ndofs = ndofs_prev*ndofs_d
    cell_ids_d = get_cell_dof_ids(space_d)

    dof_permutations_1d = TProduct._get_dof_permutation(model_d,cell_ids_d,order)

    add_dim = ndofs_prev .* collect(0:ndofs_d)
    add_dim_reorder = add_dim[dof_permutations_1d]
    node2dof_d = _tensor_product(node2dof_prev,add_dim_reorder)

    _d_dof_permutation(node2dof_d,ndofs,Val(d+1),Val(d′-1))
  end
  function _d_dof_permutation(node2dof,ndofs,::Val{d},::Val{0}) where d
    @assert d == D+1
    return node2dof
  end
  return _d_dof_permutation(Val(1),Val(D))
end

function get_tp_dof_permutation(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer;
  kwargs...) where T

  _get_tp_dof_permutation(models,spaces,order)
end

function get_tp_dof_permutation(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer;
  kwargs...) where T<:MultiValue

  @notimplemented
end

struct TProductModel{D,A,B} <: DiscreteModel{D,D}
  model::A
  models_1d::B
  function TProductModel(
    model::A,
    models_1d::B
    ) where {D,A<:CartesianDiscreteModel{D},B<:AbstractVector{<:CartesianDiscreteModel}}
    new{D,A,B}(model,models_1d)
  end
end

Geometry.get_grid(model::TProductModel) = get_grid(model.model)
Geometry.get_grid_topology(model::TProductModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::TProductModel) = get_face_labeling(model.model)

get_model(model::TProductModel) = model.model
get_1d_models(model::TProductModel) = model.models_1d

function _split_cartesian_descriptor(desc::CartesianDescriptor{D}) where D
  origin,sizes,partition,cmap,isperiodic = desc.origin,desc.sizes,desc.partition,desc.map,desc.isperiodic
  function _compute_1d_desc(
    o=first(origin.data),s=first(sizes),p=first(partition),m=cmap,i=first(isperiodic))
    CartesianDescriptor(Point(o),(s,),(p,);map=m,isperiodic=(i,))
  end
  isotropy = all([sizes[d] == sizes[1] && partition[d] == partition[1] for d = 1:D])
  factors = isotropy ? Fill(_compute_1d_desc(),D) : map(_compute_1d_desc,origin.data,sizes,partition,Fill(cmap,D),Fill(isperiodic,D))
  return factors
end

function TProductModel(args...;kwargs...)
  desc = CartesianDescriptor(args...;kwargs...)
  descs_1d = _split_cartesian_descriptor(desc)
  model = CartesianDiscreteModel(desc)
  models_1d = CartesianDiscreteModel.(descs_1d)
  TProductModel(model,models_1d)
end

struct TProductTriangulation{Dt,Dp,A,B,C} <: Triangulation{Dt,Dp}
  model::A
  trian::B
  trians_1d::C
  function TProductTriangulation(
    model::A,
    trian::B,
    trians_1d::C
    ) where {Dt,Dp,A<:TProductModel,B<:BodyFittedTriangulation{Dt,Dp},C<:AbstractVector{<:Triangulation}}
    new{Dt,Dp,A,B,C}(model,trian,trians_1d)
  end
end

Geometry.get_background_model(trian::TProductTriangulation) = trian.model
Geometry.get_grid(trian::TProductTriangulation) = get_grid(trian.trian)
Geometry.get_glue(trian::TProductTriangulation{Dt},::Val{Dt}) where Dt = get_glue(trian.trian,Dt)

function Geometry.Triangulation(model::TProductModel;kwargs...)
  trian = Triangulation(model.model;kwargs...)
  trians_1d = map(Triangulation,model.models_1d)
  TProductTriangulation(model,trian,trians_1d)
end

function CellData.get_cell_points(trian::TProductTriangulation)
  single_points = map(get_cell_points,trian.trians_1d)
  TProductCellPoint(single_points)
end

struct TProductMeasure{A,B} <: Measure
  measure::A
  measures_1d::B
end

function CellData.Measure(a::TProductTriangulation,args...;kwargs...)
  measure = Measure(a.trian,args...;kwargs...)
  measures_1d = map(Ω -> Measure(Ω,args...;kwargs...),a.trians_1d)
  TProductMeasure(measure,measures_1d)
end

CellData.get_cell_quadrature(a::TProductMeasure) = get_cell_quadrature(a.measure)

# struct TProductFESpace{A,B,C} <: SingleFieldFESpace
#   space::A
#   spaces_1d::B
#   dof_permutation::C
# end

# function TProductFESpace(
#   model::TProductModel,
#   reffe::Tuple{<:ReferenceFEName,Any,Any};
#   kwargs...)

#   basis,reffe_args,reffe_kwargs = reffe
#   T,order = reffe_args
#   cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
#   cell_reffes_1d = map(model->ReferenceFE(model,basis,T,order;reffe_kwargs...),model.models_1d)
#   space = FESpace(model,cell_reffe;kwargs...)
#   spaces_1d = map(FESpace,model.models_1d,cell_reffes_1d) # is it ok to eliminate the kwargs?
#   perm = get_tp_dof_permutation(T,model.models_1d,spaces_1d,order)
#   TProductFESpace(space,spaces_1d,perm)
# end

# FESpaces.get_triangulation(f::TProductFESpace) = get_triangulation(f.space)

# FESpaces.ConstraintStyle(::Type{<:TProductFESpace{A}}) where A = ConstraintStyle(A)

# FESpaces.get_dirichlet_dof_values(f::TProductFESpace) = get_dirichlet_dof_values(f.space)

# FESpaces.get_free_dof_ids(f::TProductFESpace) = get_free_dof_ids(f.space)

# FESpaces.get_cell_dof_ids(f::TProductFESpace) = get_cell_dof_ids(f.space)

# FESpaces.get_dirichlet_dof_ids(f::TProductFESpace) = get_dirichlet_dof_ids(f.space)

# FESpaces.num_dirichlet_tags(f::TProductFESpace) = num_dirichlet_tags(f.space)

# FESpaces.get_dirichlet_dof_tag(f::TProductFESpace) = get_dirichlet_dof_tag(f.space)

# function FESpaces.scatter_free_and_dirichlet_values(f::TProductFESpace,fv,dv)
#   scatter_free_and_dirichlet_values(f.space,fv,dv)
# end

# function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::TProductFESpace,cv)
#   gather_free_and_dirichlet_values!(fv,dv,f.space,cv)
# end

# function FESpaces.get_fe_dof_basis(f::TProductFESpace)
#   data = map(get_fe_dof_basis,f.spaces_1d)
#   trian = get_triangulation(f)
#   DS = ReferenceDomain()
#   TProductCellDof(data,trian,DS)
# end

# function FESpaces.get_fe_basis(f::TProductFESpace)
#   data = map(get_fe_basis,f.spaces_1d)
#   trian = get_triangulation(f)
#   DS = ReferenceDomain()
#   TProductCellField(data,trian,DS)
# end

# function FESpaces.get_trial_fe_basis(f::TProductFESpace)
#   data = map(get_trial_fe_basis,f.spaces_1d)
#   trian = get_triangulation(f)
#   DS = ReferenceDomain()
#   TProductCellField(data,trian,DS)
# end

# function FESpaces.get_vector_type(f::TProductFESpace)
#   D = length(f.spaces_1d)
#   V = get_vector_type(f.space)
#   T = eltype(V)
#   return TTArray{T,D}
# end

# # tensor product cell data

# struct TProductCellPoint <: CellDatum
#   data::AbstractVector
#   trian::TProductTriangulation
#   DS::DomainStyle
# end

# function Base.:(==)(a::TProductCellPoint,b::TProductCellPoint)
#   all(a.data .== b.data)
# end

# struct TProductCellField <: CellField
#   data::AbstractVector
#   trian::TProductTriangulation
#   DS::DomainStyle
# end

# CellData.get_data(a::TProductCellField) = a.data
# CellData.DomainStyle(a::TProductCellField) = a.DS
# CellData.get_triangulation(a::TProductCellField) = a.trian

# function CellData.change_domain(
#   a::TProductCellField,
#   input_domain::DomainStyle,
#   target_domain::DomainStyle)

#   b = map(data->change_domain(data,input_domain,target_domain),get_data(a))
#   trian = get_triangulation(a)
#   DS = DomainStyle(a)
#   TProductCellField(b,trian,DS)
# end

# function Arrays.evaluate!(cache,f::TProductCellField,x::TProductCellPoint)
#   @assert length(f.data) == length(x.data)
#   fx = map(evaluate!,Fill(cache,length(f.data)),f.data,x.data)
#   TProductCellArray(fx)
# end

# function CellData.integrate(f::TProductCellField,a::TProductMeasure)
#   data = map(integrate,get_data(f),a.measures_1d)
#   TProductCellArray(data)
# end

# struct TProductCellDof <: CellDatum
#   data::AbstractVector
#   trian::TProductTriangulation
#   DS::DomainStyle
# end

# CellData.get_data(a::TProductCellDof) = a.data
# CellData.DomainStyle(a::TProductCellDof) = a.DS
# CellData.get_triangulation(a::TProductCellDof) = a.trian

# (a::TProductCellDof)(f) = evaluate(a,f)

# function Arrays.evaluate!(cache,s::TProductCellDof,f::TProductCellField)
#   @assert length(s.data) == length(f.data)
#   sf = map(evaluate!,Fill(cache,length(s.data)),s.data,f.data)
#   TProductCellArray(sf)
# end

# function FESpaces.get_dof_value_type(f::TProductCellField,s::TProductCellDof)
#   fitem = first(get_data(f))
#   sitem = first(get_data(s))
#   get_dof_value_type(fitem,sitem)
# end

# struct TProductCellArray{T,N,A} <: AbstractVector{AbstractArray{T,N}}
#   data::A
#   function TProductCellArray(data::AbstractVector{<:AbstractArray{T,N}}) where {T,N}
#     A = typeof(data)
#     new{T,N,A}(data)
#   end
# end

# Base.length(a::TProductCellArray) = length(a.data)
# Base.size(a::TProductCellArray) = (length(a),)
# Base.axes(a::TProductCellArray) = (Base.OneTo(length(a)),)
# Base.getindex(a::TProductCellArray,i::Integer) = a.data[i]
# Arrays.testitem(a::TProductCellArray) = a[1]

# function Arrays.evaluate!(cache,k::Operation,a::TProductCellField...)
#   item = first(a)
#   trian = get_triangulation(item)
#   DS = DomainStyle(item)
#   @check all([get_triangulation(ai) == trian for ai in a])
#   @check all([DomainStyle(ai) == DS for ai in a])

#   D = length(get_data(item))
#   kD = Fill(k,D)

#   olddata = map(get_data,a)
#   data = map(CellData._operate_cellfields,kD,olddata...)

#   TProductCellField(data,trian,DS)
# end

# for op in (:+,:-)
#   @eval begin
#     function ($op)(a::TProductCellField,b::TProductCellField)
#       @check get_triangulation(a) == get_triangulation(b)
#       @check DomainStyle(a) == DomainStyle(b)
#       data = map($op,get_data(a),get_data(b))
#       TProductCellField(data,get_triangulation(a),DomainStyle(a))
#     end
#   end
# end

# struct GradientTProductCellField <: CellField
#   data::AbstractVector
#   gradient_data::AbstractVector
#   trian::TProductTriangulation
#   DS::DomainStyle
# end

# CellData.get_data(a::GradientTProductCellField) = a.data
# CellData.DomainStyle(a::GradientTProductCellField) = a.DS
# CellData.get_triangulation(a::GradientTProductCellField) = a.trian

# get_gradient_data(a::GradientTProductCellField) = a.gradient_data

# function CellData.gradient(a::TProductCellField)
#   data = get_data(a)
#   gradient_data = map(gradient,data)
#   trian = get_triangulation(a)
#   DS = DomainStyle(a)
#   GradientTProductCellField(data,gradient_data,trian,DS)
# end

# function Arrays.evaluate!(cache,f::GradientTProductCellField,x::TProductCellPoint)
#   @assert length(f.gradient_data) == length(x.data)
#   fx = map(evaluate!,Fill(cache,length(f.data)),f.data,x.data)
#   dfx = map(evaluate!,Fill(cache,length(f.gradient_data)),f.gradient_data,x.data)
#   GradientTProductCellArray(fx,dfx)
# end

# function CellData.evaluate!(cache,k::Operation,a::GradientTProductCellField,b::GradientTProductCellField)
#   @check get_triangulation(a) == get_triangulation(b)
#   @check DomainStyle(a) == DomainStyle(b)
#   D = length(get_data(a))
#   _cache = Fill(cache,D)
#   _k = Fill(k,D)
#   data = map(evaluate!,_cache,_k,get_data(a),get_data(b))
#   gradient_data = map(evaluate!,_cache,_k,get_gradient_data(a),get_gradient_data(b))
#   trian = get_triangulation(a)
#   DS = DomainStyle(a)
#   GradientTProductCellField(data,gradient_data,trian,DS)
# end

# function CellData.integrate(f::GradientTProductCellField,a::TProductMeasure)
#   data = map(integrate,get_data(f),a.measures_1d)
#   gradient_data = map(integrate,get_gradient_data(f),a.measures_1d)
#   GradientTProductCellArray(data,gradient_data)
# end

# struct GradientTProductCellArray{A,B}
#   data::A
#   gradient_data::B
# end
