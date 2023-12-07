struct RBIntegrationDomain
  meas::ReducedMeasure
  times::Vector{<:Real}
  idx_space::Vector{Int}
  idx_time::Vector{Int}
end

get_measure(i::RBIntegrationDomain) = i.meas
FEM.get_times(i::RBIntegrationDomain) = i.times
get_idx_space(i::RBIntegrationDomain) = i.idx_space
get_idx_time(i::RBIntegrationDomain) = i.idx_time

function common_time_integration_domain(i::Vector{RBIntegrationDomain})
  red_idx_time = map(get_idx_time,i)
  return union(red_idx_time...)
end

function get_at_time_integration_domain(
  i::Vector{RBIntegrationDomain},
  afull::PTArray,
  nparams::Int)

  idx_time = common_time_integration_domain(i)
  time_ndofs = Int(length(afull)/nparams)
  ptidx = vec(transpose(collect(0:nparams-1)*time_ndofs .+ idx_time'))
  acomm = PTArray(afull[ptidx])
  return acomm
end

function get_at_time_integration_domain(
  i::RBIntegrationDomain,
  acomm::PTArray,
  icomm::Vector{Int})

  idx_time = get_idx_time(i)
  comm_time_ndofs = length(icomm)
  nparams = Int(length(acomm)/comm_time_ndofs)
  idx_comm_to_idx_time = Int[]
  for i = eachindex(icomm)
    if icomm[i] ∈ idx_time
      push!(idx_comm_to_idx_time,first(findall(x->x==icomm[i],idx_time)))
    end
  end
  ptidx = vec(transpose(collect(0:nparams-1)*comm_time_ndofs .+ idx_comm_to_idx_time'))
  acomm = PTArray(acomm[ptidx])
  return acomm
end

function get_at_time_integration_domain(
  i::Vector{RBIntegrationDomain},
  op::PTOperator)

  nparams = length(op.μ)
  time_ndofs = length(op.tθ)
  idx_time = common_time_integration_domain(i)
  if length(idx_time) == time_ndofs
    return op
  end
  red_times = op.tθ[idx_time]
  ptidx = vec(transpose(collect(0:nparams-1)*time_ndofs .+ idx_time'))
  u0_idx = PTArray(op.u0[ptidx])
  _Us,Uts,fecache = op.ode_cache
  Us = ()
  for j in eachindex(_Us)
    spacei = _Us[j].space
    dvi = PTArray(_Us[j].dirichlet_values[ptidx])
    Us = (Us...,TrialPFESpace(dvi,spacei))
  end
  ode_cache_idx = Us,Uts,fecache
  vθ_idx = PTArray(op.vθ[ptidx])
  get_ptoperator(op.odeop,op.μ,red_times,op.dtθ,u0_idx,ode_cache_idx,vθ_idx)
end

abstract type RBAffineDecomposition{T,N} end
const RBVecAffineDecomposition{T} = RBAffineDecomposition{T,1}
const RBMatAffineDecomposition{T} = RBAffineDecomposition{T,2}

struct GenericRBAffineDecomposition{T,N} <: RBAffineDecomposition{T,N}
  basis_space::Vector{Array{T,N}}
  basis_time::Vector{Array{T}}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain
  function GenericRBAffineDecomposition(
    basis_space::Vector{Array{T,N}},
    basis_time::Vector{<:Array{T}},
    mdeim_interpolation::LU,
    integration_domain::RBIntegrationDomain) where {T,N}
    new{T,N}(basis_space,basis_time,mdeim_interpolation,integration_domain)
  end
end

istrivial(::RBAffineDecomposition) = false
get_integration_domain(a::GenericRBAffineDecomposition) = a.integration_domain

function num_rb_ndofs(a::GenericRBAffineDecomposition)
  space_ndofs = size(a.basis_space[1],1)
  time_ndofs = size(a.basis_time[2],2)
  ndofs = space_ndofs*time_ndofs
  return ndofs
end

function ReducedMeasure(a::GenericRBAffineDecomposition,trians::Triangulation...)
  if all(isnothing.(trians))
    return a
  end
  dom = get_integration_domain(a)
  meas = get_measure(dom)
  times = get_times(dom)
  idx_space = get_idx_space(dom)
  idx_time = get_idx_time(dom)
  new_meas = ReducedMeasure(meas,trians...)
  new_dom = RBIntegrationDomain(new_meas,times,idx_space,idx_time)
  return GenericRBAffineDecomposition(a.basis_space,a.basis_time,a.mdeim_interpolation,new_dom)
end

struct TrivialRBAffineDecomposition{T,N} <: RBAffineDecomposition{T,N}
  projection::Array{T,N}
end

get_projection(a::TrivialRBAffineDecomposition) = a.projection
istrivial(::TrivialRBAffineDecomposition) = true
num_rb_ndofs(a::TrivialRBAffineDecomposition) = size(get_projection(a),1)
ReducedMeasure(a::TrivialRBAffineDecomposition,args...) = a

function RBAffineDecomposition(
  rbinfo::RBInfo,
  op::PTOperator,
  nzm::NnzMatrix,
  trian::Triangulation,
  args...;
  kwargs...)

  test = op.odeop.feop.test
  meas = Measure(trian,2*get_order(test))
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
  red_times = op.tθ[interp_idx_time]
  cell_dof_ids = get_cell_dof_ids(test,trian)
  recast_interp_idx_space = recast(nzm,interp_idx_space)
  recast_interp_idx_rows,_ = vec_to_mat_idx(recast_interp_idx_space,nzm.nrows)
  red_integr_cells = get_reduced_cells(recast_interp_idx_rows,cell_dof_ids)
  red_meas = ReducedMeasure(meas,red_integr_cells)
  integr_domain = RBIntegrationDomain(red_meas,red_times,recast_interp_idx_space,interp_idx_time)
  GenericRBAffineDecomposition(proj_bs,proj_bt,lu_interp,integr_domain)
end

function RBAffineDecomposition(
  rbinfo::RBInfo,
  op::PTOperator,
  nzm::NnzMatrix{T,Affine} where T,
  trian::Triangulation,
  args...;
  kwargs...)

  nzm1 = testitem(nzm)
  projection = space_time_projection(nzm1,args...;kwargs...)
  TrivialRBAffineDecomposition(projection)
end

function get_interpolation_idx(nzm::NnzMatrix)
  get_interpolation_idx(get_nonzero_val(nzm))
end

function get_interpolation_idx(basis::Matrix)
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

function _get_at_matching_trian(
  q::Vector{<:PTArray},
  trians::Base.KeySet{Triangulation},
  trian::Triangulation)

  for (i,itrian) in enumerate(trians)
    if itrian === trian
      return q[i]
    end
  end
end

function collect_reduced_residuals!(
  cache,
  op::PTOperator,
  a::Vector{RBVecAffineDecomposition{T}}) where T

  a1 = filter(istrivial,a)
  a2 = filter(!istrivial,a)
  if !isempty(a1) && !isempty(a2)
    res1 = map(get_projection,a1)
    res2 = _collect_reduced_residuals!(cache,op,a2)
    return res1,res2
  elseif isempty(a1)
    return _collect_reduced_residuals!(cache,op,a2)
  else
    return map(get_projection,a1)
  end
end

function _collect_reduced_residuals!(
  cache,
  op::PTOperator,
  a::Vector{RBVecAffineDecomposition{T}}) where T

  b,Mcache = cache
  dom = get_integration_domain.(a)
  meas = get_measure.(dom)
  _trian = get_triangulation.(meas)
  ntrian = length(_trian)
  nparams = length(op.μ)
  icomm = common_time_integration_domain(dom)
  opt = get_at_time_integration_domain(dom,op)
  bt = get_at_time_integration_domain(dom,b,nparams)
  ress,trian = residual_for_trian!(bt,opt,op.u0,meas...)
  Mvec = Vector{Matrix{T}}(undef,ntrian)
  for j in 1:ntrian
    ress_j = _get_at_matching_trian(ress,trian,_trian[j])
    idx_space_j = get_idx_space(dom[j])
    ress_jt = get_at_time_integration_domain(dom[j],ress_j,icomm)
    setsize!(Mcache,(length(idx_space_j),length(ress_jt)))
    M = Mcache.array
    @inbounds for (n,rjt) = enumerate(ress_jt.array)
      M[:,n] = rjt[idx_space_j]
    end
    Mvec[j] = copy(M)
  end
  return Mvec
end

function collect_reduced_jacobians!(
  cache,
  op::PTOperator,
  a::Vector{RBMatAffineDecomposition{T}};
  kwargs...) where T

  a1 = filter(istrivial,a)
  a2 = filter(!istrivial,a)
  if !isempty(a1) && !isempty(a2)
    jac1 = map(get_projection,a1)
    jac2 = _collect_reduced_jacobians!(cache,op,a2;kwargs...)
    return jac1,jac2
  elseif isempty(a1)
    return _collect_reduced_jacobians!(cache,op,a2;kwargs...)
  else
    return map(get_projection,a1)
  end
end

function _collect_reduced_jacobians!(
  cache,
  op::PTOperator,
  a::Vector{RBMatAffineDecomposition{T}};
  i::Int=1) where T

  A,Mcache = cache
  dom = get_integration_domain.(a)
  meas = get_measure.(dom)
  _trian = get_triangulation.(meas)
  ntrian = length(_trian)
  nparams = length(op.μ)
  icomm = common_time_integration_domain(dom)
  opt = get_at_time_integration_domain(dom,op)
  At = get_at_time_integration_domain(dom,A,nparams)
  jacs_i,trian = jacobian_for_trian!(At,opt,opt.u0,i,meas...)
  Mvec = Vector{Matrix{T}}(undef,ntrian)
  for j in 1:ntrian
    jacs_i_j = _get_at_matching_trian(jacs_i,trian,_trian[j])
    idx_space_j = get_idx_space(dom[j])
    jacs_i_jt = get_at_time_integration_domain(dom[j],jacs_i_j,icomm)
    setsize!(Mcache,(length(idx_space_j),length(jacs_i_jt)))
    M = Mcache.array
    @inbounds for (n,jjt) = enumerate(jacs_i_jt.array)
      M[:,n] = jjt[idx_space_j].nzval
    end
    Mvec[j] = copy(M)
  end
  return Mvec
end

function rb_coefficient!(cache,a::GenericRBAffineDecomposition,b::Matrix;st_mdeim=false)
  csolve,crecast = cache
  time_ndofs = length(a.integration_domain.times)
  nparams = Int(size(b,2)/time_ndofs)
  coeff = if st_mdeim
    _coeff = mdeim_solve!(csolve,a.mdeim_interpolation,reshape(b,:,nparams))
    recast_coefficient!(crecast,first(a.basis_time),_coeff)
  else
    _coeff = mdeim_solve!(csolve,a.mdeim_interpolation,b)
    recast_coefficient!(crecast,_coeff)
  end
  return coeff
end

function rb_coefficient!(cache,::TrivialRBAffineDecomposition,args...;kwargs...)
  _,ptcache = cache
  setsize!(ptcache,(1,1))
  get_array(ptcache)
end

function mdeim_solve!(cache::CachedArray,mdeim_interpolation::LU,q::Matrix)
  setsize!(cache,size(q))
  x = cache.array
  ldiv!(x,mdeim_interpolation,q)
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

abstract type RBContributionMap <: Map end
struct RBVecContributionMap <: RBContributionMap end
struct RBMatContributionMap <: RBContributionMap end

function Arrays.return_cache(::RBVecContributionMap,snaps::PTArray{Vector{T}}) where T
  array_coeff = zeros(T,1)
  array_proj = zeros(T,1)
  CachedArray(array_coeff),CachedArray(array_proj),CachedArray(array_proj)
end

function Arrays.return_cache(::RBMatContributionMap,snaps::PTArray{Vector{T}}) where T
  array_coeff = zeros(T,1,1)
  array_proj = zeros(T,1,1)
  CachedArray(array_coeff),CachedArray(array_proj),CachedArray(array_proj)
end

function Arrays.evaluate!(
  ::RBVecContributionMap,
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
  ::RBMatContributionMap,
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
  a::RBAffineDecomposition,
  coeff::PTArray)

  basis_space_proj = a.basis_space
  basis_time = last(a.basis_time)
  map(coeff) do cn
    copy(evaluate!(k,cache,basis_space_proj,basis_time,cn))
  end
end

function rb_contribution!(
  cache,
  k::RBContributionMap,
  a::TrivialRBAffineDecomposition,
  coeff::PTArray)

  [a.projection for _ = eachindex(coeff)]
end

function zero_rb_contribution(
  ::RBVecContributionMap,
  rbinfo::RBInfo,
  rbspace::RBSpace{T}) where T

  nrow = num_rb_ndofs(rbspace)
  [zeros(T,nrow) for _ = 1:rbinfo.nsnaps_test]
end

function zero_rb_contribution(
  ::RBMatContributionMap,
  rbinfo::RBInfo,
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T}) where T

  nrow = num_rb_ndofs(rbspace_row)
  ncol = num_rb_ndofs(rbspace_col)
  [zeros(T,nrow,ncol) for _ = 1:rbinfo.nsnaps_test]
end
