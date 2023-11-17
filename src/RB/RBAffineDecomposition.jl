struct RBIntegrationDomain
  feop::PTFEOperator
  meas::Measure
  idx_space::Vector{Int}
  red_fdofs_to_fdofs::Vector{Int}
  red_ddofs_to_ddofs::Vector{Int}
  idx_time::Vector{Int}
end

function RBIntegrationDomain(
  feop::PTFEOperator,
  nzm::NnzMatrix,
  trian::TriangulationWithTags,
  idx_space::Vector{Int},
  idx_time::Vector{Int})

  test = get_test(feop)
  model = get_background_model(trian)

  recast_idx_space = recast_idx(nzm,idx_space)
  recast_idx_space_rows,_ = vec_to_mat_idx(recast_idx_space,nzm.nrows)
  cell_dof_ids = get_cell_dof_ids(test,trian)
  red_cells = get_reduced_cells(recast_idx_space_rows,cell_dof_ids)
  red_fdofs_to_fdofs = get_reduced_fdofs_to_fdofs(red_cells,cell_dof_ids)
  red_ddofs_to_ddofs = get_reduced_ddofs_to_ddofs(red_cells,cell_dof_ids)

  red_model = DiscreteModelPortion(model,red_cells)
  red_trian = if isa(trian,BoundaryTriangulationWithTags)
    BoundaryTriangulationWithTags(red_model,tags=get_tags(trian),id=get_id(trian))
  else
    TriangulationWithTags(red_model,tags=get_tags(trian),id=get_id(trian))
  end
  red_meas = Measure(red_trian,2*get_order(test))
  red_feop = reduce_fe_operator(feop,red_model)

  RBIntegrationDomain(red_feop,red_meas,recast_idx_space,red_fdofs_to_fdofs,red_ddofs_to_ddofs,idx_time)
end

function get_space_row_idx(op::PTAlgebraicOperator,rbintd::RBIntegrationDomain)
  feop = op.odeop.feop
  test = get_test(feop)
  nrows = num_free_dofs(test)
  idx_space_row, = vec_to_mat_idx(rbintd.idx_space,nrows)
  return idx_space_row
end

function reduce_cache(op::PTAlgebraicOperator,rbintd::RBIntegrationDomain)
  red_ddofs_to_ddofs = rbintd.red_ddofs_to_ddofs
  red_test = get_test(rbintd.feop)
  trials,pttrials,fecache = op.ode_cache
  red_dvals = ()
  red_trials = ()
  for trial in trials
    red_dval = map(dv->dv[red_ddofs_to_ddofs],trial.dirichlet_values)
    red_dvals = (red_dvals...,red_dval)
    red_trials = (red_trials...,PTrialFESpace(red_test,red_dval))
  end
  (red_trials,pttrials,fecache)
end

function reduce_ptoperator(op::PTAlgebraicOperator,rbintd::RBIntegrationDomain)
  red_odeop = get_algebraic_operator(rbintd.feop)
  red_ode_cache = reduce_cache(op,rbintd)
  times = op.tθ
  red_times = times[rbintd.idx_time]
  red_u0 = selectidx(op.u0,op,rbintd)
  red_vθ = selectidx(op.vθ,op,rbintd)
  get_ptoperator(red_odeop,op.μ,red_times,op.dtθ,red_u0,red_ode_cache,red_vθ)
end

function selectidx(a::PTArray,op::PTAlgebraicOperator,rbintd::RBIntegrationDomain)
  red_fdofs_to_fdofs = rbintd.red_fdofs_to_fdofs
  times = op.tθ
  red_times = times[rbintd.idx_time]
  time_to_parent_time = findall(x->x in red_times,times)
  selectidx(a,red_fdofs_to_fdofs,time_to_parent_time;nparams=length(op.μ))
end

struct RBAffineDecomposition{T,N}
  basis_space::Vector{Array{T,N}}
  basis_time::Vector{Array{T}}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain

  function RBAffineDecomposition(
    basis_space::Vector{Array{T,N}},
    basis_time::Vector{<:Array{T}},
    mdeim_interpolation::LU,
    integration_domain::RBIntegrationDomain) where {T,N}

    new{T,N}(basis_space,basis_time,mdeim_interpolation,integration_domain)
  end

  function RBAffineDecomposition(
    rbinfo::RBInfo,
    op::PTAlgebraicOperator,
    nzm::NnzMatrix,
    trian::Triangulation,
    args...;
    kwargs...)

    basis_space,basis_time = compress(nzm;ϵ=rbinfo.ϵ)
    proj_bs,proj_bt = project_space_time(basis_space,basis_time,args...;kwargs...)
    interp_idx_space = get_interpolation_idx(basis_space)
    interp_bs = basis_space[interp_idx_space,:]
    if rbinfo.st_mdeim
      interp_idx_time = get_interpolation_idx(basis_time)
      interp_bt = basis_time[interp_idx_time,:]
      interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
      lu_interp = lu(interp_bst)
    else
      interp_idx_time = collect(eachindex(op.tθ))
      lu_interp = lu(interp_bs)
    end
    integr_domain = RBIntegrationDomain(op.odeop.feop,nzm,trian,interp_idx_space,interp_idx_time)
    RBAffineDecomposition(proj_bs,proj_bt,lu_interp,integr_domain)
  end
end

const RBVecAffineDecomposition{T} = RBAffineDecomposition{T,1}
const RBMatAffineDecomposition{T} = RBAffineDecomposition{T,2}

function get_reduction_method(a::RBAffineDecomposition)
  nbs = length(a.basis_space)
  nbt = size(a.basis_time[1],2)
  lu_interp = a.mdeim_interpolation
  size(lu_interp,1) == nbs*nbt ? :spacetime : :space
end

function get_rb_ndofs(a::RBAffineDecomposition)
  space_ndofs = size(a.basis_space[1],1)
  time_ndofs = size(a.basis_time[2],2)
  ndofs = space_ndofs*time_ndofs
  return ndofs
end

function get_interpolation_idx(nzm::NnzMatrix)
  get_interpolation_idx(get_nonzero_val(nzm))
end

function get_interpolation_idx(basis::AbstractMatrix)
  n = size(basis,2)
  idx = zeros(Int,n)
  idx[1] = argmax(abs.(basis[:,1]))
  if n > 1
    @inbounds for i = 2:n
      proj = basis[:,1:i-1]*(basis[idx[1:i-1],1:i-1] \ basis[idx[1:i-1],i])
      res = basis[:,i] - proj
      idx[i] = argmax(abs.(res))
    end
  end
  unique(idx)
end

function project_space_time(basis_space::NnzMatrix,basis_time::Matrix,args...;kwargs...)
  project_space(basis_space,args...),project_time(basis_time,args...;kwargs...)
end

function project_space(
  basis_space::NnzMatrix,
  rbspace_row::RBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  compress(entire_bs_row,basis_space)
end

function project_space(
  basis_space::NnzMatrix,
  rbspace_row::RBSpace,
  rbspace_col::RBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  entire_bs_col = get_basis_space(rbspace_col)
  compress(entire_bs_row,entire_bs_col,basis_space)
end

function project_time(basis_time::Matrix,rbspace_row::RBSpace,args...;kwargs...)
  [basis_time,get_basis_time(rbspace_row)]
end

function project_time(
  basis_time::Matrix{T},
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T};
  combine_projections=(x,y)->x) where T

  bt_row = get_basis_time(rbspace_row)
  bt_col = get_basis_time(rbspace_col)
  time_ndofs = size(bt_row,1)
  nt_row = size(bt_row,2)
  nt_col = size(bt_col,2)

  bt_proj = zeros(T,time_ndofs,nt_row,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    bt_proj[:,it,jt] .= bt_row[:,it].*bt_col[:,jt]
    bt_proj_shift[2:end,it,jt] .= bt_row[2:end,it].*bt_col[1:end-1,jt]
  end

  [basis_time,combine_projections(bt_proj,bt_proj_shift)]
end

function get_reduced_cells(idx::Vector{Int},cell_dof_ids)
  get_reduced_cells(idx,Table(cell_dof_ids))
end

function get_reduced_cells(idx::Vector{Int},cell_dof_ids::Table)
  cells = Int[]
  for cell = eachindex(cell_dof_ids)
    if !isempty(intersect(idx,abs.(cell_dof_ids[cell])))
      append!(cells,cell)
    end
  end
  unique(cells)
end

function get_reduced_fdofs_to_fdofs(cells::Vector{Int},cell_dof_ids)
  get_reduced_fdofs_to_fdofs(cells,Table(cell_dof_ids))
end

function get_reduced_fdofs_to_fdofs(cells::Vector{Int},cell_dof_ids::Table)
  red_cell_dof_ids = cell_dof_ids[cells]
  red_dof_ids = red_cell_dof_ids.data
  red_free_dof_ids = red_dof_ids[findall(x->x>0,red_dof_ids)]
  red_free_dof_ids
end

function get_reduced_ddofs_to_ddofs(cells::Vector{Int},cell_dof_ids)
  get_reduced_ddofs_to_ddofs(cells,Table(cell_dof_ids))
end

function get_reduced_ddofs_to_ddofs(cells::Vector{Int},cell_dof_ids::Table)
  red_cell_dof_ids = cell_dof_ids[cells]
  red_dof_ids = red_cell_dof_ids.data
  red_dir_dof_ids = red_dof_ids[findall(x->x<0,red_dof_ids)]
  @. red_dir_dof_ids *= -1
  red_dir_dof_ids
end

function rhs_coefficient!(
  cache,
  op::PTAlgebraicOperator,
  rbres::RBVecAffineDecomposition;
  kwargs...)

  rcache,scache = cache
  red_integr_res = assemble_rhs!(rcache,op,rbres.integration_domain)
  mdeim_solve!(scache,rbres,red_integr_res;kwargs...)
end

function assemble_rhs!(
  cache,
  op::PTAlgebraicOperator,
  rbintd::RBIntegrationDomain)

  b,bmat = cache
  red_cache = selectidx(b,op,rbintd),bmat
  red_op = reduce_ptoperator(op,rbintd)
  red_meas = rbintd.meas
  red_idx = rbintd.idx_space
  collect_residuals_for_idx!(red_cache,red_op,red_idx,red_meas)
end

function lhs_coefficient!(
  cache,
  op::PTAlgebraicOperator,
  rbjac::RBMatAffineDecomposition;
  i::Int=1,kwargs...)

  jcache,scache = cache
  red_integr_jac = assemble_lhs!(jcache,op,rbjac.integration_domain;i)
  mdeim_solve!(scache,rbjac,red_integr_jac;kwargs...)
end

function assemble_lhs!(
  cache,
  op::PTAlgebraicOperator,
  rbintd::RBIntegrationDomain;
  i::Int=1)

  A,Amat = cache
  red_cache = selectidx(A,op,rbintd),Amat
  red_op = reduce_ptoperator(op,rbintd)
  red_meas = rbintd.meas
  red_idx = rbintd.idx_space
  collect_jacobians_for_idx!(red_cache,red_op,red_idx,red_meas;i)
end

function mdeim_solve!(cache,ad::RBAffineDecomposition,a::Matrix;st_mdeim=false)
  csolve,crecast = cache
  time_ndofs = length(ad.integration_domain.idx_time)
  nparams = Int(size(a,2)/time_ndofs)
  coeff = if st_mdeim
    _coeff = mdeim_solve!(csolve,ad.mdeim_interpolation,reshape(a,:,nparams))
    recast_coefficient!(crecast,first(ad.basis_time),_coeff)
  else
    _coeff = mdeim_solve!(csolve,ad.mdeim_interpolation,a)
    recast_coefficient!(crecast,_coeff)
  end
  return coeff
end

function mdeim_solve!(cache::CachedArray,mdeim_interp::LU,q::Matrix)
  setsize!(cache,size(q))
  x = cache.array
  ldiv!(x,mdeim_interp,q)
  x
end

function recast_coefficient!(
  cache::PTArray{<:CachedArray{T}},
  coeff::Matrix{T}) where T

  Qs = Int(size(coeff,1))
  nparams = length(cache)
  Nt = Int(size(coeff,2)/nparams)
  setsize!(cache,(Nt,Qs))
  ptarray = get_array(cache)

  @inbounds for n = eachindex(ptarray)
    ptarray[n] .= coeff[:,(n-1)*Nt+1:n*Nt]'
  end

  ptarray
end

function recast_coefficient!(
  cache::PTArray{<:CachedArray{T}},
  basis_time::Matrix{T},
  coeff::Matrix{T}) where T

  Nt,Qt = size(basis_time)
  Qs = Int(size(coeff,1)/Qt)
  setsize!(cache,(Nt,Qs))
  ptarray = get_array(cache)

  @inbounds for n = axes(coeff,2)
    an = ptarray[n]
    cn = coeff[:,n]
    for qs in 1:Qs
      sorted_idx = [(i-1)*Qs+qs for i = 1:Qt]
      an[:,qs] .= basis_time*cn[sorted_idx]
    end
  end

  ptarray
end

abstract type RBContributionMap{T} <: Map end
struct RBVecContributionMap{T} <: RBContributionMap{T}
  RBVecContributionMap(::Type{T}) where T = new{T}()
end
struct RBMatContributionMap{T} <: RBContributionMap{T}
  RBMatContributionMap(::Type{T}) where T = new{T}()
end

function Arrays.return_cache(::RBVecContributionMap{T}) where T
  array_coeff = zeros(T,1)
  array_proj = zeros(T,1)
  CachedArray(array_coeff),CachedArray(array_proj),CachedArray(array_proj)
end

function Arrays.return_cache(::RBMatContributionMap{T}) where T
  array_coeff = zeros(T,1,1)
  array_proj = zeros(T,1,1)
  CachedArray(array_coeff),CachedArray(array_proj),CachedArray(array_proj)
end

function Arrays.evaluate!(
  ::RBVecContributionMap{T},
  cache,
  proj_basis_space::AbstractVector,
  basis_time::Matrix{T},
  coeff::Matrix{T}) where T

  @assert length(proj_basis_space) == size(coeff,2)
  proj1 = testitem(proj_basis_space)

  cache_coeff,cache_proj,cache_proj_global = cache
  num_rb_times = size(basis_time,2)
  num_rb_dofs = length(proj1)*size(basis_time,2)
  setsize!(cache_coeff,(num_rb_times,))
  setsize!(cache_proj,(num_rb_dofs,))
  setsize!(cache_proj_global,(num_rb_dofs,))

  array_coeff = cache_coeff.array
  array_proj = cache_proj.array
  array_proj_global = cache_proj_global.array
  array_proj_global .= zero(T)

  @inbounds for i = axes(coeff,2)
    array_coeff .= basis_time'*coeff[:,i]
    LinearAlgebra.kron!(array_proj,proj_basis_space[i],array_coeff)
    array_proj_global .+= array_proj
  end

  array_proj_global
end

function Arrays.evaluate!(
  ::RBMatContributionMap{T},
  cache,
  proj_basis_space::AbstractVector,
  basis_time::Array{T,3},
  coeff::Matrix{T}) where T

  @assert length(proj_basis_space) == size(coeff,2)
  proj1 = testitem(proj_basis_space)

  cache_coeff,cache_proj,cache_proj_global = cache
  num_rb_times_row = size(basis_time,2)
  num_rb_times_col = size(basis_time,3)
  num_rb_rows = size(proj1,1)*size(basis_time,2)
  num_rb_cols = size(proj1,2)*size(basis_time,3)
  setsize!(cache_coeff,(num_rb_times_row,num_rb_times_col))
  setsize!(cache_proj,(num_rb_rows,num_rb_cols))
  setsize!(cache_proj_global,(num_rb_rows,num_rb_cols))

  array_coeff = cache_coeff.array
  array_proj = cache_proj.array
  array_proj_global = cache_proj_global.array
  array_proj_global .= zero(T)

  for i = axes(coeff,2)
    for col in 1:num_rb_times_col
      for row in 1:num_rb_times_row
        @fastmath @inbounds array_coeff[row,col] = sum(basis_time[:,row,col].*coeff[:,i])
      end
    end
    LinearAlgebra.kron!(array_proj,proj_basis_space[i],array_coeff)
    array_proj_global .+= array_proj
  end

  array_proj_global
end

function rb_contribution!(
  cache,
  k::RBContributionMap,
  ad::RBAffineDecomposition,
  coeff::PTArray)

  basis_space_proj = ad.basis_space
  basis_time = last(ad.basis_time)
  map(coeff) do cn
    copy(evaluate!(k,cache,basis_space_proj,basis_time,cn))
  end
end
