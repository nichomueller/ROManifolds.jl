function allocate_sys_cache(
  feop::PTFEOperator,
  rbspace::RBSpace,
  snaps::PTArray{T}) where T

  fe_ndofs = get_num_free_dofs(feop.test)
  nsnaps = length(snaps)
  x = PTArray([zeros(T,fe_ndofs) for _ = 1:nsnaps])
  b = allocate_residual(feop,x)
  A = allocate_jacobian(feop,x)
  rb_ndofs = get_rb_ndofs(rbspace)
  xrb = PTArray([zeros(T,rb_ndofs) for _ = 1:nsnaps])
  brb = allocate_residual(rbspace,xrb)
  Arb = allocate_jacobian(rbspace,xrb)
  res_cache = CachedArray(b),CachedArray(brb),CachedArray(xrb),CachedArray(xrb)
  jac_cache = CachedArray(A),CachedArray(Arb),CachedArray(xrb),CachedArray(xrb)
  sol_cache = CachedArray(xrb)
  res_cache,jac_cache,sol_cache
end

function reduced_order_model(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver)

  # Offline phase
  if info.load_structures
    sols,params,rbspace,rbrhs,rblhs = load(info,(
      AbstractSnapshots,
      Table,
      AbstractRBSpace,
      AbstractRBAlgebraicContribution,
      AbstractRBAlgebraicContribution))
  end
  nsnaps = info.nsnaps_state
  params = realization(feop,nsnaps)
  sols = collect_solutions(fesolver,feop,params)
  rbspace = get_reduced_basis(info,feop,sols,fesolver,params)
  rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
  save(info,(sols,params,rbspace,rbrhs,rblhs))

  # Online phase
  nsnaps = info.nsnaps_online
  sols_test,params_test = load_test(info,feop,fesolver,nsnaps)
  rb_results = test_rb_solver(info,feop,fesolver,rbspace,rbrhs,rblhs,sols,sols_test,params_test)
  save(info,rb_results)

  return
end

function test_rb_solver(
  info::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PODESolver,
  rbspace::AbstractRBSpace,
  rbres::AbstractRBAlgebraicContribution{T},
  rbjacs::AbstractRBAlgebraicContribution{T},
  ::Snapshots,
  snaps_test::Snapshots,
  params_test::Table) where T

  printstyled("Solving linear RB problems\n";color=:blue)
  sys_cache = allocate_system_cache(info,feop,fesolver,rbspace,rbres)
  rhs = collect_rhs_contributions(sys_cache,info,feop,fesolver,rbres,rbspace,params_test)
  lhs = collect_lhs_contributions(sys_cache,info,feop,fesolver,rbjacs,rbspace,params_test)

  wall_time = @elapsed begin
    rb_snaps_test = solve(fesolver,rbspace,rhs,lhs)
  end
  approx_snaps_test = recast(rbspace,rb_snaps_test)
  RBResults(info,feop,snaps_test,approx_snaps_test,wall_time)
end

function test_rb_solver(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::AbstractRBSpace,
  rbres::AbstractRBAlgebraicContribution{T},
  rbjacs::AbstractRBAlgebraicContribution{T},
  snaps::PTArray,
  snaps_test::PTArray,
  params_test::Table) where T

  printstyled("Solving nonlinear RB problems with Newton iterations\n";color=:blue)
  sys_cache = allocate_system_cache(info,feop,fesolver,rbspace,rbres)
  nl_cache = nothing
  x = initial_guess(snaps,params_test)
  _,conv0 = Algebra._check_convergence(fesolver.nls,x)
  iter = 0
  wall_time = @elapsed begin
    for iter in 1:fesolver.nls.max_nliters
      rhs = collect_rhs_contributions!(sys_cache,info,feop,fesolver,rbres,rbspace,params_test,x)
      lhs = collect_lhs_contributions!(sys_cache,info,feop,fesolver,rbjacs,rbspace,params_test,x)
      nl_cache = solve!(x,fesolver,rhs,lhs,nl_cache)
      x .-= recast(rbspace,x)
      isconv,conv = Algebra._check_convergence(fesolver.nls,x,conv0)
      println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv))) \n")
      if all(isconv); return; end
      if iter == nls.max_nliters
        @unreachable
      end
    end
  end
  RBResults(info,feop,snaps_test,x,wall_time)
end

function Algebra.solve(fesolver::PODESolver,rhs::PTArray,lhs::PTArray)
  x = copy(rhs)
  cache = nothing
  solve!(x,fesolver,rhs,lhs,cache)
end

function Algebra.solve!(
  x::PTArray,
  fesolver::PODESolver,
  rhs::PTArray,
  lhs::PTArray,
  ::Nothing)

  lhsaff,rhsaff = Nonaffine(),Nonaffine()
  ss = symbolic_setup(fesolver.nls.ls,testitem(lhs))
  ns = numerical_setup(ss,lhs,lhsaff)
  _loop_solve!(x,ns,rhs,lhsaff,rhsaff)
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbres::RBAlgebraicContribution{T},
  args...) where T

  coeff_cache,red_cache = cache
  nmeas = num_domains(rbres)
  meas = get_domains(rbres)
  st_mdeim = info.st_mdeim

  rb_res_contribs = Vector{PTArray{Matrix{T}}}(undef,nmeas)
  for m in meas
    rbresm = rbres[m]
    coeff = rhs_coefficient!(coeff_cache,feop,fesolver,rbresm,meas,args...;st_mdeim)
    contrib = rb_contribution!(red_cache,rbresm,coeff)
    push!(rb_res_contribs,contrib)
  end
  return sum(rb_res_contribs)
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
  b::PTArray,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbres::RBAffineDecomposition,
  meas::Base.KeySet{Measure},
  sols::PTArray,
  μ::Table)

  idx = rbres.integration_domain.idx
  collect_residual!(b,fesolver,feop,sols,μ,meas)
  map(x->getindex(x,idx),b)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjacs::Vector{AbstractRBAlgebraicContribution{T}},
  args...) where T

  njacs = length(rbjacs)
  rb_jacs_contribs = Vector{<:PTArray{Matrix{T}}}(undef,nmeas)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    rb_jacs_contribs[i] = collect_lhs_contributions!(cache,info,feop,fesolver,rb_jac_i,args...;i)
  end
  return sum(rb_jacs_contribs)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjac::RBAlgebraicContribution{T},
  args...;
  kwargs...) where T

  coeff_cache,red_cache = cache
  nmeas = num_domains(rbjac)
  meas = get_domains(rbjac)
  st_mdeim = info.st_mdeim

  rb_jac_contribs = Vector{PTArray{Matrix{T}}}(undef,nmeas)
  for m in 1:meas
    rbjacm = rbjac[m]
    coeff = lhs_coefficient!(coeff_cache,feop,fesolver,rbjacm,meas,args...;st_mdeim,kwargs...)
    contrib = rb_contribution!(red_cache,rbjacm,coeff)
    push!(rb_jac_contribs,contrib)
  end
  return sum(rb_jac_contribs)
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
  A::PTArray,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbjac::RBAffineDecomposition,
  meas::Base.KeySet{Measure},
  input...;
  i::Int=1)

  idx = rbjac.integration_domain.idx
  collect_jacobian!(A,fesolver,feop,sols,μ,meas;i)
  map(x->getindex(x,idx),A)
end

function mdeim_solve!(cache,ad::RBAffineDecomposition,b::PTArray;st_mdeim=false)
  if st_mdeim
    coeff = mdeim_solve!(coeff,ad.mdeim_interpolation,reshape(b,:))
    recast_coefficient!(ad.basis_time,coeff)
  else
    mdeim_solve!(ad.mdeim_interpolation,b)
  end
end

function mdeim_solve!(cache::PTArray,mdeim_interp::LU,b::PTArray)
  setsize!(cache,size(testitem(b)))
  ldiv!(x,mdeim_interp,b)
  x_t = map(transpose,x)
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

  @fastmath @inbounds for (ic,c) in enumerate(eachcol(coeff))
    @fastmath @inbounds for (row,b) in enumerate(eachcol(bt_proj))
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

  @fastmath @inbounds for (ic,c) in enumerate(eachcol(coeff))
    @fastmath @inbounds for col in axes(bt_proj,3), row in axes(bt_proj,2)
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
  sz = map(*,size(ad.basis_space),size(coeff))
  setsize!(cache,sz)
  @inbounds for n = eachindex(coeff)
    rn = cache[n].array
    cn = coeff[n]
    Threads.@threads for i = eachindex(coeff)
      LinearAlgebra.kron!(rn,ad.basis_space[i],cn[i])
    end
  end

  PTArray(map(x->getproperty(x,:array)),rcoeff.array)
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
