struct RBIntegrationDomain
  trian::Triangulation
  times::Vector{<:Real}
  idx::Vector{Int}

  function RBIntegrationDomain(trian::Triangulation,times::Vector{<:Real},idx::Vector{Int})
    new(trian,times,idx)
  end

  function RBIntegrationDomain(
    feop::PTFEOperator,
    trian::Triangulation,
    times::Vector{<:Real},
    interp_idx_space::Vector{Int},
    interp_idx_time::Vector{Int},
    entire_interp_idx_space::Vector{Int};
    st_mdeim=false)

    cell_dof_ids = get_cell_dof_ids(feop.test,trian)
    red_integr_cells = find_cells(entire_interp_idx_space,cell_dof_ids)
    red_trian = view(trian,red_integr_cells)
    red_times = st_mdeim ? times[interp_idx_time] : times
    RBIntegrationDomain(red_trian,red_times,interp_idx_space)
  end
end

function Arrays.testvalue(::Type{RBIntegrationDomain},feop::PTFEOperator)
  test = get_test(feop)
  trian = get_triangulation(test)
  times = Vector{Real}(undef,0)
  idx = Vector{Int}(undef,0)
  RBIntegrationDomain(trian,times,idx)
end

struct RBAffineDecomposition{T}
  basis_space::Vector{Array{T}}
  basis_time::Array{T}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain

  function RBAffineDecomposition(
    basis_space::Vector{<:Array{T}},
    basis_time::Array{T},
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
    proj_bs,proj_bt = compress(basis_space,basis_time,args...;kwargs...)
    interp_idx_space,interp_idx_time = get_interpolation_idx(basis_space,basis_time)
    entire_interp_idx_space = recast_idx(basis_space,interp_idx_space)

    interp_bs = basis_space.nonzero_val[interp_idx_space,:]
    lu_interp = if info.st_mdeim
      interp_bt = basis_time.nonzero_val[interp_idx_time,:]
      interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
      lu(interp_bst)
    else
      lu(interp_bs)
    end

    integr_domain = RBIntegrationDomain(
      feop,
      trian,
      times,
      interp_idx_space,
      interp_idx_time,
      entire_interp_idx_space;
      info.st_mdeim)

    RBAffineDecomposition(proj_bs,proj_bt,lu_interp,integr_domain)
  end
end

function Arrays.testvalue(
  ::Type{RBAffineDecomposition{T}},
  feop::PTFEOperator) where T

  basis_space = Vector{Matrix{T}}(undef,0)
  basis_time = Matrix{T}(undef,0)
  mdeim_interpolation = lu(one(T))
  integration_domain = testvalue(IntegrationDomain,feop)
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
  ress,trian = collect_residuals(fesolver,feop,snaps,μ;return_trian=true)
  ad_res = compress_component(info,feop,ress,trian,times,rbspace)
  return ad_res
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::RBSpace,
  snaps::PTArray,
  μ::Table)

  times = get_times(fesolver)
  θ = fesolver.θ

  njacs = length(feop.jacs)
  ad_jacs = Vector{RBAlgebraicContribution}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : x-y
    jacs,trian = collect_jacobians(fesolver,feop,snaps,μ;i,return_trian=true)
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

function get_interpolation_idx(nzm::NnzMatrix...)
  get_interpolation_idx.(get_nonzero_val.(nzm))
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

function compress(basis_space::NnzMatrix,::NnzMatrix,args...;kwargs...)
  compress_space(basis_space,args...),compress_time(args...;kwargs...)
end

function compress_space(
  basis_space::NnzMatrix,
  rbspace_row::RBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  compress(entire_bs_row,basis_space)
end

function compress_space(
  basis_space::NnzMatrix,
  rbspace_row::RBSpace,
  rbspace_col::RBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  entire_bs_col = get_basis_space(rbspace_col)
  compress(entire_bs_row,entire_bs_col,basis_space)
end

function compress_time(rbspace_row::RBSpace,args...;kwargs...)
  get_basis_time(rbspace_row)
end

function compress_time(
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

  combine_projections(bt_proj,bt_proj_shift)
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
  rbspace::RBSpace,
  args...;
  kwargs...)

  rcache,ccache,pcache = cache
  red_integr_res = assemble_rhs!(rcache,feop,fesolver,rbres,args...)
  coeff = mdeim_solve!(ccache,rbres,red_integr_res;kwargs...)
  project_rhs_coefficient!(pcache,rbspace,rbres.basis_time,coeff)
end

function assemble_rhs!(
  cache::PTArray,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbres::RBAffineDecomposition,
  trian::Base.KeySet{Triangulation},
  sols::PTArray,
  μ::Table)

  ndofs = num_free_dofs(feop.test)
  setsize!(cache,(ndofs,))
  b = get_array(cache)

  red_idx = rbres.integration_domain.idx
  red_times = rbres.integration_domain.times
  red_trian = rbres.integration_domain.trian
  strian = substitute_trian(red_trian,trian)
  meas = map(t->get_measure(feop,t),strian)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,red_times)

  collect_residuals!(b,fesolver,ode_op,sols,μ,ode_cache,red_trian,meas...)
  map(x->getindex(x,red_idx),b)
end

function lhs_coefficient!(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjac::RBAffineDecomposition,
  args...;
  i::Int=1,kwargs...)

  jcache,ccache,pcache = cache
  red_integr_jac = assemble_lhs!(jcache,feop,fesolver,rbjac,args...;i)
  coeff = mdeim_solve!(ccache,rbjac,red_integr_jac;kwargs...)
  project_lhs_coefficient!(pcache,rbjac.basis_time,coeff)
end

function assemble_lhs!(
  cache::PTArray,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbjac::RBAffineDecomposition,
  trian::Base.KeySet{Triangulation},
  input...;
  i::Int=1)

  ndofs_row = num_free_dofs(feop.test)
  ndofs_col = num_free_dofs(get_trial(feop)(nothing,nothing))
  setsize!(cache,(ndofs_row,ndofs_col))
  A = get_array(cache)

  red_idx = rbjac.integration_domain.idx
  red_times = rbjac.integration_domain.times
  red_trian = rbjac.integration_domain.trian
  strian = substitute_trian(red_trian,trian)
  meas = map(t->get_measure(feop,t),strian)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,red_times)

  collect_jacobians!(A,fesolver,ode_op,sols,μ,ode_cache,red_trian,meas...;i)
  map(x->getindex(x,red_idx),A)
end

function mdeim_solve!(cache,ad::RBAffineDecomposition,b::PTArray;st_mdeim=false)
  if st_mdeim
    coeff = mdeim_solve!(cache,ad.mdeim_interpolation,reshape(b,:))
    recast_coefficient!(cache,ad.basis_time,coeff)
  else
    mdeim_solve!(cache,ad.mdeim_interpolation,b)
  end
end

function mdeim_solve!(cache::PTArray,mdeim_interp::LU,b::PTArray)
  setsize!(cache,size(testitem(b)))
  x = map(y->ldiv!(y,mdeim_interp,b),cache)
  map(transpose,x)
end

function recast_coefficient!(
  rcoeff::PTArray{<:CachedArray{T}},
  basis_time::Vector{Matrix{T}},
  coeff::PTArray) where T

  bt,_ = basis_time
  Nt,Qt = size(bt)
  Qs = Int(length(testitem(coeff))/Qt)
  setsize!(rcoeff,Nt,Qs)

  @inbounds for n = eachindex(coeff)
    rn = rcoeff[n].array
    cn = coeff[n]
    for qs in 1:Qs
      sorted_idx = [(i-1)*Qs+qs for i = 1:Qt]
      copyto!(view(rn,:,qs),bt*cn[sorted_idx])
    end
  end

  PTArray(map(x->getproperty(x,:array)),rcoeff.array)
end

function project_rhs_coefficient(
  basis_time::Vector{<:Array{T}},
  coeff::AbstractMatrix) where T

  _,bt_proj = basis_time
  nt_row = size(bt_proj,2)
  Qs = size(coeff,2)
  pcoeff = zeros(T,nt_row,1)
  pcoeff_v = Vector{typeof(pcoeff)}(undef,Qs)

  @inbounds for (ic,c) in enumerate(eachcol(coeff))
    for (row,b) in enumerate(eachcol(bt_proj))
      pcoeff[row] = sum(b.*c)
    end
    pcoeff_v[ic] = pcoeff
  end

  pcoeff_v
end

function project_lhs_coefficient(
  basis_time::Vector{<:Array{T}},
  coeff::AbstractMatrix) where T

  _,bt_proj = basis_time
  nt_row,nt_col = size(bt_proj)[2:3]
  Qs = size(coeff,2)
  pcoeff = zeros(T,nt_row,nt_col)
  pcoeff_v = Vector{typeof(pcoeff)}(undef,Qs)

  @inbounds for (ic,c) in enumerate(eachcol(coeff))
    for col in axes(bt_proj,3), row in axes(bt_proj,2)
      pcoeff[row,col] = sum(bt_proj[:,row,col].*c)
    end
    pcoeff_v[ic] = pcoeff
  end

  pcoeff_v
end

function rb_contribution!(
  cache::PTArray{<:CachedArray{T}},
  ad::RBAffineDecomposition,
  coeff::PTArray{T}) where T

  bs = ad.basis_space
  sz = map(*,size(bs),size(coeff))
  setsize!(cache,sz)
  @inbounds for n = eachindex(coeff)
    rn = cache[n].array
    cn = coeff[n]
    Threads.@threads for i = eachindex(coeff)
      LinearAlgebra.kron!(rn,bs[i],cn[i])
    end
  end

  PTArray(map(x->getproperty(x,:array)),rcoeff.array)
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
      combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : x-y
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
