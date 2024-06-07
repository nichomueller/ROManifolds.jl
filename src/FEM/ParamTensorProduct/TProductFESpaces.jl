function get_comp_to_free_dofs(::Type{T},space::FESpace,cell_reffe) where T
  @abstractmethod
end

function get_comp_to_free_dofs(::Type{T},space::UnconstrainedFESpace,cell_reffe) where T
  glue = space.metadata
  ncomps = num_components(T)
  free_dof_to_comp = if isnothing(glue)
    _get_free_dof_to_comp(space,cell_reffe)
  else
    glue.free_dof_to_comp
  end
  get_comp_to_free_dofs(free_dof_to_comp,ncomps)
end

function _get_free_dof_to_comp(space,cell_reffe)
  reffe = testitem(cell_reffe)
  ldof_to_comp = get_dof_to_comp(reffe)
  cell_dof_ids = get_cell_dof_ids(space)
  nfree = num_free_dofs(space)
  dof_to_comp = zeros(eltype(ldof_to_comp),nfree)
  @inbounds for dofs_cell in cell_dof_ids
    for (ldof,dof) in enumerate(dofs_cell)
      if dof > 0
        dof_to_comp[dof] = ldof_to_comp[ldof]
      end
    end
  end
  return dof_to_comp
end

function get_comp_to_free_dofs(dof2comp,ncomps)
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

function _get_terms(p::Polytope,orders)
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  terms = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  return terms
end

function _get_dof_permutation(
  model::CartesianDiscreteModel{Dc},
  cell_dof_ids::Table{Ti},
  order::Integer) where {Dc,Ti}

  desc = get_cartesian_descriptor(model)

  periodic = desc.isperiodic
  ncells = desc.partition
  ndofs = order .* ncells .+ 1 .- periodic

  terms = _get_terms(first(get_polytopes(model)),fill(order,Dc))
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  new_dof_ids = LinearIndices(ndofs)
  n2o_dof_map = fill(Ti(-1),ndofs)

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
  model::CartesianDiscreteModel{D},
  space::UnconstrainedFESpace,
  order::Integer,
  args...
  ) where {T,D}

  cell_dof_ids = get_cell_dof_ids(space)
  dof_perm = _get_dof_permutation(model,cell_dof_ids,order)
  return IndexMap(dof_perm)
end

function _get_dof_permutation(
  ::Type{T},
  model::CartesianDiscreteModel{D},
  space::UnconstrainedFESpace,
  order::Integer,
  comp_to_dofs::AbstractVector
  ) where {T<:MultiValue,D}

  cell_dof_ids = get_cell_dof_ids(space)
  Ti = eltype(eltype(cell_dof_ids))
  dof_perms = Array{Ti,D}[]
  for dofs in comp_to_dofs
    cell_dof_comp_ids = _get_cell_dof_comp_ids(cell_dof_ids,dofs)
    dof_perm_comp = _get_dof_permutation(model,cell_dof_comp_ids,order)
    push!(dof_perms,dof_perm_comp)
  end
  return MultiValueIndexMap(dof_perms)
end

function get_dof_permutation(args...)
  index_map = _get_dof_permutation(args...)
  free_dofs_map(index_map)
end

# this function computes only the free dofs tensor product permutation
function _get_tp_dof_permutation(models::AbstractVector,spaces::AbstractVector,order::Integer)
  @assert length(models) == length(spaces)
  D = length(models)

  function _tensor_product(aprev::AbstractArray{T,M},a::AbstractVector) where {T,M}
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

function _get_tp_dof_permutation(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer
  ) where T

  dof_perm,dof_perm_1d = _get_tp_dof_permutation(models,spaces,order)
  return IndexMap(dof_perm),dof_perm_1d
end

function _get_tp_dof_permutation(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer
  ) where T<:MultiValue

  ncomp = num_components(T)
  dof_perm,dof_perms_1d = _get_tp_dof_permutation(eltype(T),models,spaces,order)
  ncomp_dof_perm = compose_indices(dof_perm,ncomp)
  return ncomp_dof_perm,dof_perms_1d
end

function get_tp_dof_permutation(args...)
  dof_perm,dof_perms_1d = _get_tp_dof_permutation(args...)
  return TProductIndexMap(dof_perm,dof_perms_1d)
end

function _compute_dof_permutations(::Type{T},model,space::FESpace,spaces_1d,cell_reffe,order) where T
  @abstractmethod
end

function _compute_dof_permutations(::Type{T},model,space::UnconstrainedFESpace,spaces_1d,cell_reffe,order) where T
  comp_to_dofs = get_comp_to_free_dofs(T,space,cell_reffe)
  dof_permutation = get_dof_permutation(T,model.model,space,order,comp_to_dofs)
  tp_dof_permutation = get_tp_dof_permutation(T,model.models_1d,spaces_1d,order)
  return dof_permutation,tp_dof_permutation
end

function _compute_dof_permutations(::Type{T},model,z::ZeroMeanFESpace,spaces_1d,cell_reffe,order) where T
  uspace = z.space.space
  dof_to_fix = z.space.dof_to_fix
  _dof_permutation,_tp_dof_permutation = _compute_dof_permutations(T,model,uspace,spaces_1d,cell_reffe,order)
  dof_permutation = FixedDofIndexMap(_dof_permutation,dof_to_fix)
  tp_dof_permutation = FixedDofIndexMap(_dof_permutation,findfirst(vec(_tp_dof_permutation).==dof_to_fix))
  return dof_permutation,tp_dof_permutation
end

function univariate_spaces(
  model::TProductModel,
  cell_reffes;
  dirichlet_tags=Int[],
  conformity=nothing,
  vector_type=nothing,
  kwargs...)

  add_1d_tags!(model,dirichlet_tags)
  map((model,cell_reffe) -> FESpace(model,cell_reffe;dirichlet_tags,conformity,vector_type),
    model.models_1d,cell_reffes)
end

function _get_tt_vector(f,perm::AbstractIndexMap)
  V = get_vector_type(f)
  T = eltype(V)
  vec = allocate_vector(TTVector{T},perm)
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
  cell_reffes_1d = map(model->ReferenceFE(model,basis,eltype(T),order;reffe_kwargs...),model.models_1d)
  space = FESpace(model.model,cell_reffe;kwargs...)
  spaces_1d = univariate_spaces(model,cell_reffes_1d;kwargs...)
  dof_permutation,tp_dof_permutation = _compute_dof_permutations(T,model,space,spaces_1d,cell_reffe,order)
  TProductFESpace(space,spaces_1d,dof_permutation,tp_dof_permutation)
end

abstract type TProductStyle end
struct TProduct <: TProductStyle end
struct UnTProduct <: TProductStyle end

TProductStyle(f::F) where F = TProductStyle(F)
TProductStyle(::Type{<:FESpace}) = @abstractmethod
TProductStyle(::Type{<:SingleFieldFESpace}) = UnTProduct()
TProductStyle(::Type{<:TProductFESpace}) = TProduct()
for F in (:TrialFESpace,:TransientTrialFESpace,:TrialParamFESpace,:FESpaceToParamFESpace,:TransientTrialParamFESpace)
  @eval begin
    TProductStyle(::Type{<:$F{<:TProductFESpace}}) = TProduct()
  end
end

is_tensor_product(f::F) where F = (TProductStyle(F)==TProduct())
is_tensor_product(::Type{F}) where F = (TProductStyle(F)==TProduct())

get_space(f::TProductFESpace) = f.space

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

# need to correct free dof values for fe spaces employing the TT strategy

for F in (:TrialFESpace,:TransientTrialFESpace,:TransientTrialParamFESpace)
  @eval begin
    FESpaces.zero_free_values(f::$F{<:TProductFESpace}) = zero_free_values(f.space)
  end
end

for F in (:TrialParamFESpace,:FESpaceToParamFESpace)
  @eval begin
    function FESpaces.zero_free_values(f::$F{<:TProductFESpace})
      V = get_vector_type(f)
      vector = zero_free_values(f.space)
      array_of_similar_arrays(vector,length(V))
    end
  end
end

function FESpaces.zero_free_values(f::MultiFieldFESpace{<:TProductFESpace})
  V = get_vector_type(f)
  f′ = MultiFieldFESpace(V,map(get_space,f.spaces))
  vector = zero_free_values(f′)
end

function FESpaces.zero_free_values(f::MultiFieldParamFESpace{<:TProductFESpace})
  V = get_vector_type(f)
  f′ = MultiFieldParamFESpace(V,map(get_space,f.spaces))
  vector = zero_free_values(f′)
  array_of_similar_arrays(vector,length(V))
end

# need to correct parametric, tproduct, zeromean constrained fespaces
function FESpaces.FEFunction(
  f::FESpaceToParamFESpace{<:TProductFESpace{D,<:ZeroMeanFESpace},L},
  free_values::ParamArray,
  dirichlet_values::ParamArray) where {D,L}
  FEFunction(FESpaceToParamFESpace(f.space.space,Val{L}()),free_values,dirichlet_values)
end

function FESpaces.EvaluationFunction(
  f::FESpaceToParamFESpace{<:TProductFESpace{D,<:ZeroMeanFESpace},L},
  free_values::ParamArray) where {D,L}
  EvaluationFunction(FESpaceToParamFESpace(f.space.space,Val{L}()),free_values)
end

for F in (:TrialFESpace,:ZeroMeanFESpace,:FESpaceWithConstantFixed)
  @eval begin
    function FESpaces.scatter_free_and_dirichlet_values(
      f::FESpaceToParamFESpace{<:TProductFESpace{D,<:$F},L},
      fv::ParamArray,
      dv::ParamArray) where {D,L}
      scatter_free_and_dirichlet_values(FESpaceToParamFESpace(f.space.space,Val{L}()),fv,dv)
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
Base.length(a::TProductFEBasis) = length(get_data(a))

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

function get_sparse_index_map(U::TProductFESpace,V::TProductFESpace)
  sparsity = get_sparsity(U,V)
  psparsity = permute_sparsity(sparsity,U,V)
  I,J,_ = findnz(psparsity)
  i,j,_ = univariate_findnz(psparsity)
  g2l = global_2_local_nnz(psparsity,I,J,i,j)
  pg2l = permute_index_map(psparsity,g2l,U,V)
  return SparseIndexMap(pg2l,psparsity)
end

function global_2_local_nnz(sparsity::TProductSparsityPattern,I,J,i,j)
  IJ = get_nonzero_indices(sparsity)
  lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

  unrows = univariate_num_rows(sparsity)
  uncols = univariate_num_cols(sparsity)
  unnz = univariate_nnz(sparsity)
  g2l = zeros(eltype(IJ),unnz...)

  @inbounds for (k,gid) = enumerate(IJ)
    irows = Tuple(tensorize_indices(I[k],unrows))
    icols = Tuple(tensorize_indices(J[k],uncols))
    iaxes = CartesianIndex.(irows,icols)
    lid = map((i,j) -> findfirst(i.==[j]),lids,iaxes)
    g2l[lid...] = gid
  end

  return g2l
end

function _permute_index_map(perm,I,J)
  nrows = length(I)
  IJ = vec(I) .+ nrows .* (vec(J)'.-1)
  iperm = copy(perm)
  @inbounds for (k,pk) in enumerate(perm)
    iperm[k] = IJ[pk]
  end
  return IndexMap(iperm)
end

function permute_index_map(::TProductSparsityPattern,perm,U::TProductFESpace,V::TProductFESpace)
  I = get_dof_permutation(V)
  J = get_dof_permutation(U)
  return _permute_index_map(perm,I,J)
end

function permute_index_map(
  sparsity::TProductSparsityPattern{<:MultiValuePatternCSC},
  perm,U::TProductFESpace,V::TProductFESpace)

  function _to_component_indices(i,ncomps,icomp)
    nrows = Int(num_free_dofs(V)/ncomps)
    ic = copy(i)
    @inbounds for (j,IJ) in enumerate(i)
      I = fast_index(IJ,nrows)
      J = slow_index(IJ,nrows)
      I′ = (I-1)*ncomps + icomp
      J′ = (J-1)*ncomps + icomp
      ic[j] = (J′-1)*nrows*ncomps + I′
    end
    return ic
  end

  I = get_dof_permutation(V)
  J = get_dof_permutation(U)
  I1 = get_component(I,1;multivalue=false)
  J1 = get_component(J,1;multivalue=false)
  indices = _permute_index_map(perm,I1,J1)
  ncomps = num_components(sparsity)
  indices′ = map(icomp->_to_component_indices(indices,ncomps,icomp),1:ncomps)
  return MultiValueIndexMap(indices′)
end
