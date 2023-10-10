struct RBIntegrationDomain
  meas::Measure
  times::Vector{<:Real}
  idx::Vector{Int}
end

function Arrays.testvalue(::Type{RBIntegrationDomain},feop::PTFEOperator)
  test = get_test(feop)
  trian = get_triangulation(test)
  meas = get_measure(feop,trian)
  times = Vector{Real}(undef,0)
  idx = Vector{Int}(undef,0)
  RBIntegrationDomain(meas,times,idx)
end

struct RBAffineDecomposition{T}
  basis_space::Vector{Array{T}}
  basis_time::Vector{Array{T}}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain

  function RBAffineDecomposition(
    basis_space::Vector{<:Array{T}},
    basis_time::Vector{<:Array{T}},
    mdeim_interpolation::LU,
    integration_domain::RBIntegrationDomain) where T
    new{T}(basis_space,basis_time,mdeim_interpolation,integration_domain)
  end

  function RBAffineDecomposition(
    info::RBInfo,
    feop::PTFEOperator,
    nzm::NnzMatrix,
    trian::Triangulation,
    times::Vector{<:Real},
    args...;
    kwargs...)

    basis_space,basis_time = compress(nzm;ϵ=info.ϵ)
    proj_bs,proj_bt = project_space_time(basis_space,basis_time,args...;kwargs...)
    interp_idx_space = get_interpolation_idx(basis_space)
    interp_idx_time = get_interpolation_idx(basis_time)
    entire_interp_idx_space = recast_idx(nzm,interp_idx_space)
    entire_interp_idx_rows,_ = vec_to_mat_idx(entire_interp_idx_space,nzm.nrows)

    interp_bs = basis_space[interp_idx_space,:]
    lu_interp = if info.st_mdeim
      interp_bt = basis_time[interp_idx_time,:]
      interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
      lu(interp_bst)
    else
      lu(interp_bs)
    end

    cell_dof_ids = get_cell_dof_ids(feop.test,trian)
    red_integr_cells = find_cells(entire_interp_idx_rows,cell_dof_ids)
    red_trian = view(trian,red_integr_cells)
    red_meas = get_measure(feop,red_trian)
    red_times = info.st_mdeim ? times[interp_idx_time] : times
    integr_domain = RBIntegrationDomain(red_meas,red_times,entire_interp_idx_space)

    RBAffineDecomposition(proj_bs,proj_bt,lu_interp,integr_domain)
  end
end

function Arrays.testvalue(
  ::Type{RBAffineDecomposition{T}},
  feop::PTFEOperator;
  vector=true) where T

  if vector
    basis_space = [zeros(T,1)]
    basis_time = [zeros(T,1,1),zeros(T,1,1)]
  else
    basis_space = [zeros(T,1,1)]
    basis_time = [zeros(T,1,1),zeros(T,1,1,1)]
  end
  mdeim_interpolation = lu(one(T))
  integration_domain = testvalue(RBIntegrationDomain,feop)
  RBAffineDecomposition(basis_space,basis_time,mdeim_interpolation,integration_domain)
end

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::RBSpace,
  snaps::Snapshots,
  μ::Table)

  nsnaps = info.nsnaps_system
  snapsθ = recenter(fesolver,snaps,μ)
  _snapsθ,_μ = snapsθ[1:nsnaps],μ[1:nsnaps]
  rhs = collect_compress_rhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  lhs = collect_compress_lhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  rhs,lhs
end

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::RBSpace,
  snaps::PTArray,
  μ::Table)

  times = get_times(fesolver)
  ress,trian = collect_residuals_for_trian(fesolver,feop,snaps,μ,times)
  ad_res = compress_component(info,feop,ress,trian,times,rbspace)
  return ad_res
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::RBSpace{T},
  snaps::PTArray,
  μ::Table) where T

  times = get_times(fesolver)
  θ = fesolver.θ

  njacs = length(feop.jacs)
  ad_jacs = Vector{RBAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    jacs,trian = collect_jacobians_for_trian(fesolver,feop,snaps,μ,times;i)
    ad_jacs[i] = compress_component(info,feop,jacs,trian,times,rbspace,rbspace;combine_projections)
  end
  return ad_jacs
end

function compress_component(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::Vector{NnzMatrix{T}},
  trian::Base.KeySet{Triangulation},
  args...;
  kwargs...) where T

  contrib = RBAlgebraicContribution(T)
  for (i,ti) in enumerate(trian)
    si = snaps[i]
    ci = RBAffineDecomposition(info,feop,si,ti,args...;kwargs...)
    add_contribution!(contrib,ti,ci)
  end
  contrib
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
  basis_time::Matrix,
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

function find_cells(idx::Vector{Int},cell_dof_ids)
  find_cells(Val{length(idx)>length(cell_dof_ids)}(),idx,cell_dof_ids)
end

function find_cells(::Val{true},idx::Vector{Int},cell_dof_ids)
  cells = Int[]
  for cell = eachindex(cell_dof_ids)
    if !isempty(intersect(idx,abs.(cell_dof_ids[cell])))
      append!(cells,cell)
    end
  end
  unique(cells)
end

function find_cells(::Val{false},idx::Vector{Int},cell_dof_ids)
  cells = Vector{Int}[]
  for i = idx
    cell = findall(x->!isempty(intersect(abs.(x),i)),cell_dof_ids)
    cells = isempty(cell) ? cells : push!(cells,cell)
  end
  unique(reduce(vcat,cells))
end

function rhs_coefficient!(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbres::RBAffineDecomposition,
  args...;
  kwargs...)

  rcache,scache... = cache
  red_integr_res = assemble_rhs!(rcache,feop,fesolver,rbres,args...)
  mdeim_solve!(scache,rbres,red_integr_res;kwargs...)
end

function assemble_rhs!(
  cache::PTArray,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbres::RBAffineDecomposition,
  sols::PTArray,
  μ::Table)

  ndofs = num_free_dofs(feop.test)
  setsize!(cache,(ndofs,))

  red_idx = rbres.integration_domain.idx
  red_times = rbres.integration_domain.times
  red_meas = rbres.integration_domain.meas

  b = get_array(cache;len=length(red_times)*length(μ))
  sols = get_solutions_at_times(sols,fesolver,red_times)

  collect_residuals_for_idx!(b,fesolver,feop,sols,μ,red_times,red_idx,red_meas)
end

function lhs_coefficient!(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjac::RBAffineDecomposition,
  args...;
  i::Int=1,kwargs...)

  jcache,scache... = cache
  red_integr_jac = assemble_lhs!(jcache,feop,fesolver,rbjac,args...;i)
  mdeim_solve!(scache,rbjac,red_integr_jac;kwargs...)
end

function assemble_lhs!(
  cache::PTArray,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbjac::RBAffineDecomposition,
  sols::PTArray,
  μ::Table;
  i::Int=1)

  ndofs_row = num_free_dofs(feop.test)
  ndofs_col = num_free_dofs(get_trial(feop)(nothing,nothing))
  setsize!(cache,(ndofs_row,ndofs_col))

  red_idx = rbjac.integration_domain.idx
  red_times = rbjac.integration_domain.times
  red_meas = rbjac.integration_domain.meas

  A = get_array(cache;len=length(red_times)*length(μ))
  sols = get_solutions_at_times(sols,fesolver,red_times)

  collect_jacobians_for_idx!(A,fesolver,feop,sols,μ,red_times,red_idx,red_meas;i)
end

function mdeim_solve!(cache,ad::RBAffineDecomposition,a::Matrix;st_mdeim=false)
  csolve,crecast = cache
  time_ndofs = length(ad.integration_domain.times)
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

function get_solutions_at_times(sols::PTArray,fesolver::PODESolver,red_times::Vector{<:Real})
  times = get_times(fesolver)
  time_ndofs = length(times)
  nparams = Int(length(sols)/time_ndofs)
  if length(red_times) < time_ndofs
    tidx = findall(x->x in red_times,times)
    ptidx = reshape(transpose(collect(0:nparams-1)*time_ndofs .+ tidx'),:)
    PTArray(sols[ptidx])
  else
    sols
  end
end

struct RBContributionMap <: Map end

function Arrays.return_cache(
  ::RBContributionMap,
  proj_basis_space::AbstractVector,
  basis_time::Matrix{T}) where T

  proj1 = testitem(proj_basis_space)

  num_rb_times = size(basis_time,2)
  num_rb_dofs = length(proj1)*size(basis_time,2)
  array_coeff = zeros(T,num_rb_times)
  array_proj = zeros(T,num_rb_dofs)
  CachedArray(array_coeff),CachedArray(array_proj),CachedArray(array_proj)
end

function Arrays.return_cache(
  ::RBContributionMap,
  proj_basis_space::AbstractVector,
  basis_time::Array{T,3}) where T

  proj1 = testitem(proj_basis_space)

  num_rb_times_row = size(basis_time,2)
  num_rb_times_col = size(basis_time,3)
  num_rb_rows = size(proj1,1)*size(basis_time,2)
  num_rb_cols = size(proj1,2)*size(basis_time,3)
  array_coeff = zeros(T,num_rb_times_row,num_rb_times_col)
  array_proj = zeros(T,num_rb_rows,num_rb_cols)
  CachedArray(array_coeff),CachedArray(array_proj),CachedArray(array_proj)
end

function Arrays.evaluate!(
  ::RBContributionMap,
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
  ::RBContributionMap,
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
  ad::RBAffineDecomposition,
  coeff::PTArray{T}) where T

  basis_space_proj = ad.basis_space
  basis_time = last(ad.basis_time)
  k = RBContributionMap()
  map(coeff) do cn
    copy(evaluate!(k,cache,basis_space_proj,basis_time,cn))
  end
end

# Multifield interface

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  times::Vector{<:Real},
  rbspace::BlockRBSpace...;
  kwargs...)

  nfields = get_nfields(testitem(rbspace))
  times = get_times(fesolver)
  contrib = testvalue(RBBlockAlgebraicContribution{T},feop,(nfields,1))
  @inbounds for i_field = index_pairs(nfields,1)
    row,_ = i_field
    feop_i = filter_operator(feop,i_field)
    rbspace_i = map(x->filter_rbspace(x,row),rbspace)
    ress_i,meas_i = collect_residuals(feop_i,fesolver,args...)
    if iszero(ress_i)
      contrib.touched[row,1] = false
    else
      contrib.block[row,1] = compress_component(info,feop_i,ress_i,meas_i,times,rbspace_i)
    end
  end
  return contrib
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  times::Vector{<:Real},
  rbspace::BlockRBSpace...;
  kwargs...)

  njacs = length(feop.jacs)
  nfields = get_nfields(testitem(rbspace))
  times = get_times(fesolver)
  θ = fesolver.θ
  contrib = testvalue(RBBlockAlgebraicContribution{T},feop,(nfields,nfields))
  contribs = Vector{typeof(contrib)}(undef,njacs)
  @inbounds for i_field = index_pairs(nfields,nfields)
    row,col = i_field
    feop_i = filter_operator(feop,i_field)
    rbspace_i = map(x->filter_rbspace(x,i_field),rbspace)
    for i = 1:njacs
      combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
      jacs_i,meas_i = collect_jacobians(feop,fesolver,args...;i)
      if iszero(ress_i)
        contribs[i].touched[row,col] = false
      else
        contribs.block[row,col] = compress_component(
          info,
          feop_i,
          jacs_i,
          meas_i,
          times,
          rbspace_i...;
          combine_projections)
      end
    end
  end
  return contribs
end

# Multifield interface
# function rhs_coefficient!(
#   cache,
#   feop::PTFEOperator,
#   fesolver::PODESolver,
#   rbres::BlockRBAffineDecomposition,
#   rbspace::BlockRBSpace,
#   args...;
#   kwargs...)

#   nfields = get_nfields(rbres)
#   @inbounds for (row,col) in index_pairs(nfields,1)
#     if rbres.touched[row,col]
#       rhs_coefficient!(cache,feop,fesolver,rbres[row,col],args...;kwargs...)
#     else
#       nrows = get_spacetime_ndofs(rbspace[row])
#       ncols = get_spacetime_ndofs(rbspace[col])
#       zero_rhs_coeff(rbres)
#     end
#   end
# end
# function collect_rhs_contributions!(
#   cache,
#   info::RBInfo,
#   feop::PTFEOperator,
#   fesolver::PODESolver,
#   rbres::BlockRBAffineDecomposition,
#   rbspace::BlockRBSpace,
#   args...) where T

#   nfields = get_nfields(rbres)
#   rb_res_contribs = Vector{<:PTArray{Matrix{T}}}(undef,nmeas)
#   @inbounds for (row,col) in index_pairs(nfields,1)
#     if rbres.touched[row,col]
#       collect_rhs_contributions!(cache,feop,fesolver,rbres[row,col],rbspace[row],args...;kwargs...)
#     else
#       nrows = get_spacetime_ndofs(rbspace[row])
#       ncols = get_spacetime_ndofs(rbspace[col])
#       zero_rhs_coeff(rbres)
#     end
#   end
#   return sum(rb_res_contribs)
# end
