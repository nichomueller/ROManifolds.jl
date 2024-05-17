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

abstract type AbstractIndexMap{D} <: AbstractArray{Int,D} end

struct IndexMap{D} <: AbstractIndexMap{D}
  indices::Array{Int,D}
end

Base.size(i::IndexMap) = size(i.indices)
Base.getindex(i::IndexMap,j...) = getindex(i.indices,j...)

function free_dofs_map(i::IndexMap)
  free_dofs_locations = findall(i.indices.>0)
  IndexMapView(i.indices,free_dofs_locations)
end

function dirichlet_dofs_map(i::IndexMap)
  dir_dofs_locations = findall(i.indices.<0)
  i.indices[dir_dofs_locations]
end

struct IndexMapView{D,L} <: AbstractIndexMap{D}
  indices::Array{Int,D}
  locations::L
end

Base.size(i::IndexMapView) = _shape_per_dir(i.locations)
Base.IndexStyle(::Type{<:IndexMapView}) = IndexLinear()
Base.getindex(i::IndexMapView,j::Int) = i.indices[i.locations[j]]

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

  terms = _get_terms(first(get_polytopes(model)),fill(order,Dc))
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  new_dof_ids = LinearIndices(ndofs)
  n2o_dof_map = fill(-1,ndofs)

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
  dof_perm = _get_dof_permutation(model,cell_dof_ids,order)
  return IndexMap(dof_perm)
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
  return IndexMap(dof_perm)
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

    dof_permutations_1d = _get_dof_permutation(model_d,cell_ids_d,order)

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

function get_inv_tp_dof_permutation(args...;kwargs...)
  tp_dof_permutation = get_tp_dof_permutation(args...;kwargs...)
  invp = invperm(vec(tp_dof_permutation))
  inv_tp_dof_permutation = reshape(invp,size(tp_dof_permutation))
  IndexMap(inv_tp_dof_permutation)
end

function _get_tt_vector(f,perm::AbstractIndexMap{D}) where D
  V = get_vector_type(f)
  T = eltype(V)
  vec = allocate_vector(TTVector{D,T},perm)
  return vec
end

function _get_tt_vector_type(f,perm)
  typeof(_get_tt_vector(f,perm))
end

struct TProductFESpace{D,A<:SingleFieldFESpace,B<:AbstractVector{<:SingleFieldFESpace},V} <: SingleFieldFESpace
  space::A
  spaces_1d::B
  dof_permutation::IndexMap{D}
  inv_tp_dof_permutation::IndexMap{D}
  vector_type::Type{V}
end

function TProductFESpace(
  space::SingleFieldFESpace,
  spaces_1d::Vector{<:SingleFieldFESpace},
  dof_permutation::AbstractIndexMap,
  inv_tp_dof_permutation::AbstractIndexMap)

  vector_type = _get_tt_vector_type(space,dof_permutation)
  TProductFESpace(space,spaces_1d,dof_permutation,inv_tp_dof_permutation,vector_type)
end

function FESpaces.FESpace(
  model::TProductModel,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args
  cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
  cell_reffes_1d = map(model->ReferenceFE(model,basis,T,order;reffe_kwargs...),model.models_1d)
  space = FESpace(model.model,cell_reffe;kwargs...)
  spaces_1d = univariate_spaces(model,cell_reffes_1d;kwargs...)
  dof_permutation = get_dof_permutation(T,model.model,space,order)
  inv_tp_dof_permutation = get_inv_tp_dof_permutation(T,model.models_1d,spaces_1d,order)
  TProductFESpace(space,spaces_1d,dof_permutation,inv_tp_dof_permutation)
end

function univariate_spaces(model::TProductModel,cell_reffes;dirichlet_tags=Int[],kwargs...)
  add_1d_tags!(model,dirichlet_tags)
  map((model,cell_reffe) -> FESpace(model,cell_reffe;dirichlet_tags,kwargs...),
    model.models_1d,cell_reffes)
end

get_dof_permutation(f::TProductFESpace) = f.dof_permutation

get_free_dof_permutation(f::TProductFESpace) = free_dofs_map(f.dof_permutation)

FESpaces.get_triangulation(f::TProductFESpace) = get_triangulation(f.space)

FESpaces.get_free_dof_ids(f::TProductFESpace) = get_free_dof_ids(f.space)

function FESpaces.zero_free_values(f::TProductFESpace)
  vec = _get_tt_vector(f.space,get_free_dof_permutation(f))
  fill!(vec,zero(eltype(vec)))
end

FESpaces.get_vector_type(f::TProductFESpace) = f.vector_type

FESpaces.get_dof_value_type(f::TProductFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::TProductFESpace) = get_cell_dof_ids(f.space)

FESpaces.ConstraintStyle(::Type{<:TProductFESpace{D,A}}) where {D,A} = ConstraintStyle(A)

FESpaces.get_fe_basis(f::TProductFESpace) = get_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::TProductFESpace) = get_fe_dof_basis(f.space)

FESpaces.num_dirichlet_dofs(f::TProductFESpace) = num_dirichlet_dofs(f.space)

FESpaces.get_cell_isconstrained(f::TProductFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::TProductFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::TProductFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::TProductFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_tags(f::TProductFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::TProductFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::TProductFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

get_dirichlet_cells(f::TProductFESpace) = get_dirichlet_cells(f.space)

# need to correct free dof values for trial spaces defined for TT problems

for F in (:TrialFESpace,:TransientTrialFESpace)
  @eval begin
    FESpaces.zero_free_values(f::$F{<:TProductFESpace}) = zero_free_values(f.space)
  end
end

for F in (:TrialParamFESpace,:FESpaceToParamFESpace,:TransientTrialParamFESpace)
  @eval begin
    function FESpaces.zero_free_values(f::$F{<:TProductFESpace})
      V = get_vector_type(f)
      vector = zero_free_values(f.space)
      allocate_param_array(vector,length(V))
    end
  end
end

struct TProductFEBasis{DS,BS,A,B} <: FEBasis
  basis::A
  trian::B
  domain_style::DS
  basis_style::BS
end

function TProductFEBasis(basis::Vector,trian::TProductTriangulation)
  b1 = testitem(basis)
  DS = DomainStyle(b1)
  BS = BasisStyle(b1)
  @check all(map(i -> DS===DomainStyle(i) && BS===BasisStyle(i),basis))
  TProductFEBasis(basis,trian,DS,BS)
end

CellData.get_data(f::TProductFEBasis) = f.basis
CellData.get_triangulation(f::TProductFEBasis) = f.trian
FESpaces.BasisStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = BS
CellData.DomainStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = DS
MultiField.num_fields(a::TProductFEBasis) = length(get_data(a))
Base.length(a::TProductFEBasis) = num_fields(a)

function get_tp_triangulation(f::TProductFESpace)
  trian = get_triangulation(f.space)
  trians_1d = map(get_triangulation,f.spaces_1d)
  TProductTriangulation(trian,trians_1d)
end

function get_tp_fe_basis(f::TProductFESpace)
  basis = map(get_fe_basis,f.spaces_1d)
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

function get_tp_trial_fe_basis(f::TProductFESpace)
  basis = map(get_trial_fe_basis,f.spaces_1d)
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

get_tp_trial_fe_basis(f::TrialFESpace{<:TProductFESpace}) = get_tp_trial_fe_basis(f.space)

function assemble_norm_matrix(f,U::FESpace,V::FESpace)
  assemble_matrix(f,U,V)
end

function assemble_norm_matrix(f,U::TProductFESpace,V::TProductFESpace)
  a = SparseMatrixAssembler(U,V)
  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(U)
  assemble_matrix(a,collect_cell_matrix(U,V,f(u,v)))
end

function assemble_norm_matrix(f,U::TrialFESpace{<:TProductFESpace},V::TProductFESpace)
  assemble_norm_matrix(f,U.space,V)
end
