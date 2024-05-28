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

function _get_dof_permutation(
  ::Type{T},
  model::CartesianDiscreteModel,
  space::UnconstrainedFESpace,
  order::Integer;
  kwargs...) where T

  cell_dof_ids = get_cell_dof_ids(space)
  dof_perm = _get_dof_permutation(model,cell_dof_ids,order)
  return IndexMap(dof_perm)
end

function _get_dof_permutation(
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

function get_dof_permutation(args...;kwargs...)
  index_map = _get_dof_permutation(args...;kwargs...)
  free_dofs_map(index_map)
end

# this function computes only the free dofs tensor product permutation
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
  function _local_dof_permutation(model,space)
    cell_ids = get_cell_dof_ids(space)
    dof_permutations_1d = _get_dof_permutation(model,cell_ids,order)
    free_dof_permutations_1d = dof_permutations_1d[findall(dof_permutations_1d.>0)]
    return free_dof_permutations_1d
  end

  free_dof_permutations_1d = map(_local_dof_permutation,models,spaces)

  function _d_dof_permutation(::Val{1},::Val{d′}) where d′
    @assert d′ == D
    space_d = spaces[1]
    ndofs_d = num_free_dofs(space_d)
    ndofs = ndofs_d
    free_dof_permutation_1d = free_dof_permutations_1d[1]
    return _d_dof_permutation(free_dof_permutation_1d,ndofs,Val(2),Val(d′-1))
  end
  function _d_dof_permutation(node2dof_prev,ndofs_prev,::Val{d},::Val{d′}) where {d,d′}
    space_d = spaces[d]
    ndofs_d = num_free_dofs(space_d)
    ndofs = ndofs_prev*ndofs_d
    free_dof_permutation_1d = free_dof_permutations_1d[d]

    add_dim = ndofs_prev .* collect(0:ndofs_d)
    add_dim_reorder = add_dim[free_dof_permutation_1d]
    node2dof_d = _tensor_product(node2dof_prev,add_dim_reorder)

    _d_dof_permutation(node2dof_d,ndofs,Val(d+1),Val(d′-1))
  end
  function _d_dof_permutation(node2dof,ndofs,::Val{d},::Val{0}) where d
    @assert d == D+1
    return node2dof
  end
  return _d_dof_permutation(Val(1),Val(D)),free_dof_permutations_1d
end

function get_tp_dof_permutation(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer;
  kwargs...) where T

  dof_perm,dof_perms_1d = _get_tp_dof_permutation(models,spaces,order)
  return TProductIndexMap(dof_perm,dof_perms_1d)
end

function get_tp_dof_permutation(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer;
  kwargs...) where T<:MultiValue

  @notimplemented
end

function univariate_spaces(model::TProductModel,cell_reffes;dirichlet_tags=Int[],kwargs...)
  add_1d_tags!(model,dirichlet_tags)
  map((model,cell_reffe) -> FESpace(model,cell_reffe;dirichlet_tags,kwargs...),
    model.models_1d,cell_reffes)
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

struct TProductFESpace{D,A,B,I1,I2,V} <: SingleFieldFESpace
  space::A
  spaces_1d::B
  dof_permutation::I1
  tp_dof_permutation::I2
  vector_type::Type{V}
  function TProductFESpace(
    space::A,
    spaces_1d::B,
    dof_permutation::I1,
    tp_dof_permutation::I2,
    vector_type::Type{V}
    ) where {D,A,B,I1<:AbstractIndexMap{D},I2<:AbstractIndexMap{D},V}

    new{D,A,B,I1,I2,V}(space,spaces_1d,dof_permutation,tp_dof_permutation,vector_type)
  end
end

function TProductFESpace(
  space::SingleFieldFESpace,
  spaces_1d::Vector{<:SingleFieldFESpace},
  dof_permutation::AbstractIndexMap,
  tp_dof_permutation::AbstractIndexMap)

  vector_type = _get_tt_vector_type(space,dof_permutation)
  TProductFESpace(space,spaces_1d,dof_permutation,tp_dof_permutation,vector_type)
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
  tp_dof_permutation = get_tp_dof_permutation(T,model.models_1d,spaces_1d,order)
  TProductFESpace(space,spaces_1d,dof_permutation,tp_dof_permutation)
end

get_dof_permutation(f::TProductFESpace) = f.dof_permutation

get_tp_dof_permutation(f::TProductFESpace) = f.tp_dof_permutation

for F in (:TrialFESpace,:TransientTrialFESpace,:TrialParamFESpace,:FESpaceToParamFESpace,:TransientTrialParamFESpace)
  @eval begin
    get_dof_permutation(f::$F{<:TProductFESpace}) = get_dof_permutation(f.space)
  end
end

FESpaces.get_triangulation(f::TProductFESpace) = get_triangulation(f.space)

FESpaces.get_free_dof_ids(f::TProductFESpace) = get_free_dof_ids(f.space)

function FESpaces.zero_free_values(f::TProductFESpace)
  vec = _get_tt_vector(f.space,get_dof_permutation(f))
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

for F in (:TrialFESpace,:TransientTrialFESpace,:TrialParamFESpace,:FESpaceToParamFESpace,:TransientTrialParamFESpace)
  @eval begin
    function get_sparsity(U::$F{<:TProductFESpace},V::TProductFESpace)
      get_sparsity(U.space,V)
    end

    function get_sparse_index_map(U::$F{<:TProductFESpace},V::TProductFESpace)
      get_sparse_index_map(U.space,V)
    end
  end
end

function get_sparsity(U::TProductFESpace,V::TProductFESpace)
  a = SparseMatrixAssembler(U,V)
  sparsity = get_sparsity(a.assem,U.space,V.space)
  sparsities_1d = map(get_sparsity,a.assems_1d,U.spaces_1d,V.spaces_1d)
  return TProductSparsityPattern(sparsity,sparsities_1d)
end

function permute_sparsity(s::SparsityPattern,U::FESpace,V::FESpace)
  psparsity = permute_sparsity(s.sparsity,U.space,V.space)
  psparsities = map(permute_sparsity,s.sparsities_1d,U.spaces_1d,V.spaces_1d)
  TProductSparsityPattern(psparsity,psparsities)
end

function permute_sparsity(s::TProductSparsityPattern,U::TProductFESpace,V::TProductFESpace)
  index_map_I = get_dof_permutation(V)
  index_map_J = get_dof_permutation(U)
  index_map_I_1d = get_tp_dof_permutation(V).indices_1d
  index_map_J_1d = get_tp_dof_permutation(U).indices_1d
  permute_sparsity(s,(index_map_I,index_map_I_1d),(index_map_J,index_map_J_1d))
end

function permute_ids(i::AbstractArray{Int},perm::AbstractArray{Int})
  ip = copy(i)
  @inbounds for (k,ik) in enumerate(ip)
    ip[k] = perm[ik]
  end
  return ip
end

function get_sparse_index_map(U::TProductFESpace,V::TProductFESpace)
  sparsity = get_sparsity(U,V)
  psparsity = permute_sparsity(sparsity,U,V)
  I,J,_ = findnz(psparsity)
  i,j,_ = univariate_findnz(psparsity)
  pg2l = _global_2_local_nnz(psparsity,I,J,i,j)
  g2l = _invperm(pg2l,U,V)
  return SparseIndexMap(g2l,psparsity)
end

function _global_2_local_nnz(sparsity,I,J,i,j)
  IJ = get_nonzero_indices(sparsity)
  lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

  unrows = univariate_num_rows(sparsity)
  uncols = univariate_num_cols(sparsity)
  unnz = univariate_nnz(sparsity)
  g2l = zeros(Int,unnz...)

  @inbounds for (k,gid) = enumerate(IJ)
    irows = Tuple(tensorize_indices(I[k],unrows))
    icols = Tuple(tensorize_indices(J[k],uncols))
    iaxes = CartesianIndex.(irows,icols)
    global2local = map((i,j) -> findfirst(i.==[j]),lids,iaxes)
    g2l[global2local...] = gid
  end

  return IndexMap(g2l)
end

function _invperm(perm,U::TProductFESpace,V::TProductFESpace)
  nrows = num_free_dofs(V)
  index_map_I = vec(get_dof_permutation(V))
  index_map_J = vec(get_dof_permutation(U))
  index_map_IJ = index_map_I .+ nrows .* (index_map_J'.-1)
  iperm = copy(perm)
  @inbounds for (k,pk) in enumerate(perm)
    iperm[k] = index_map_IJ[pk]
  end
  return IndexMap(iperm)
end

# function recast_indices(indices::AbstractVector,f::FESpace...)
#   return indices
# end

# function recast_indices(indices::AbstractVector,V::TProductFESpace)
#   p = get_dof_permutation(V)
#   return vec(p)[indices]
# end

# function recast_indices(indices::AbstractVector,U::TProductFESpace,V::TProductFESpace)
#   nrows = num_free_dofs(V)
#   pU = get_dof_permutation(U)
#   pV = get_dof_permutation(V)
#   pUV = vec(pV) .+ nrows .* (vec(pU)'.-1)
#   return vec(pUV)[indices]
# end

# for F in (:TrialFESpace,:TransientTrialFESpace,:TrialParamFESpace,:FESpaceToParamFESpace,:TransientTrialParamFESpace)
#   @eval begin
#     function recast_indices(indices::AbstractVector,U::$F{<:TProductFESpace},V::TProductFESpace)
#       recast_indices(indices,U.space,V)
#     end
#   end
# end
# for F in (:TrialFESpace,:TransientTrialFESpace,:TrialParamFESpace,:FESpaceToParamFESpace,:TransientTrialParamFESpace)
#   @eval begin
#     function recast_indices(indices::AbstractVector,U::$F{<:TProductFESpace},V::TProductFESpace)
#       @warn "deactivated recast indices for jacobians during debugging"
#       return indices
#     end
#   end
# end
