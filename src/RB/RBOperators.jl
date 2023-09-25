struct RBResidualMap end
struct RBJacobianMap end

function Arrays.return_cache(
  ::RBResidualMap,
  rb_res::RBAlgebraicContribution{T};
  st_mdeim=)

  nmeas = num_domains(rb_res)
  coeff = zeros(T,)
  array_coeff = Vector{Matrix{T}}(undef,nmeas)
  array_lhs = Vector{Matrix{T}}(undef,nmeas)
  array_rhs = Vector{Matrix{T}}(undef,nmeas)
  @inbounds for i = 1:nmeas
    array_coeff[i] = zeros(T)
  end
end

function reduce_fe_operator(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver)

  # Offline phase
  nsnaps = info.nsnaps_state
  params = realization(feop,nsnaps)
  sols = collect_solutions(feop,fesolver,params)
  rbspace = get_reduced_basis(info,feop,sols,fesolver,params)

  rb_res = collect_compress_residual(info,feop,fesolver,rbspace,sols,params)
  rb_jacs = collect_compress_jacobians(info,feop,fesolver,rbspace,sols,params)

  save(info,(sols,params,rbspace))

  # Online phase
  nsnaps = info.nsnaps_online
  sols,params = load_test(info,feop,fesolver,nsnaps)
  online_res = collect_residual_contributions(info,feop,fesolver,rb_res,sols,params)
  online_jac = collect_jacobians_contributions(info,feop,fesolver,rb_jacs,sols,params)
  rb_results = test_rb_solver(online_res,online_jac,rbspace,sols,params)
  save(info,rb_results)

  return
end

function collect_residual_contributions(
  info::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PODESolver,
  rb_res::RBAlgebraicContribution{T},
  args...) where T

  meas = get_domains(rb_res)
  st_mdeim = info.st_mdeim

  rb_res_contribs = Matrix{T}[]
  for m in meas
    coeff = residual_coefficient(feop,fesolver,meas,args...;st_mdeim)
    rb_res_contrib = rb_contribution(rb_res[m],coeff)
    push!(rb_res_contribs,rb_res_contrib)
  end
  return sum(rb_res_contribs)
end

function collect_jacobian_contributions(
  info::RBInfo,
  feop::PTFEOperator{Affine},
  fesolver::PODESolver,
  rb_jacs::Vector{RBAlgebraicContribution{T}},
  args...) where T

  st_mdeim = info.st_mdeim

  rb_jacs_contribs = map(enumerate(rb_jacs)) do i
    rb_jac_i = rb_jacs[i]
    meas = get_domains(rb_jacs_i)
    rb_jac_contribs = Matrix{T}[]
    for m in meas
      coeff = jacobian_contribution(feop,fesolver,meas,args...;st_mdeim)
      rb_jac_contrib = rb_contribution(rb_jac_i[m],coeff)
      push!(rb_jac_contribs,rb_jac_contrib)
    end
    sum(rb_jac_contribs)
  end
  return sum(rb_jacs_contribs)
end

function residual_coefficient(
  feop::PTFEOperator,
  fesolver::PODESolver,
  res_ad::RBAffineDecomposition,
  args...;
  kwargs...)

  red_integr_res = assemble_residual(feop,fesolver,res_ad,args...)
  coeff = mdeim_solve(res_ad,red_integr_res;kwargs...)
  project_residual_coefficient(res_ad.basis_time,coeff)
end

function jacobian_coefficient(
  feop::PTFEOperator,
  fesolver::PODESolver,
  jac_ad::RBAffineDecomposition,
  args...;
  i::Int=1,kwargs...)

  red_integr_jac = assemble_jacobian(feop,fesolver,jac_ad,args...;i)
  coeff = mdeim_solve(jac_ad,red_integr_jac;kwargs...)
  project_jacobian_coefficient(jac_ad.basis_time,coeff)
end

function assemble_residual(
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  res_ad::RBAffineDecomposition,
  meas::Vector{Measure},
  input...)

  idx = res_ad.integration_domain.idx


  collect_residual(fesolver,feop,snaps,args...)
  res_cat = reduce(hcat,res)
  res_cat[idx,:]
end

function assemble_jacobian(
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  jac_ad::RBAffineDecomposition,
  measures::Vector{Measure},
  input...;
  i::Int=1)

  idx = jac_ad.integration_domain.idx
  meas = jac_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)

  collector = CollectJacobiansMap(fesolver,feop,trian,new_meas...;i)
  jac = collector.f(input...)
  jac_cat = hcat(jac...)
  jac_cat.nonzero_val[idx]
end

function mdeim_solve(ad::RBAffineDecomposition,b::AbstractArray;st_mdeim=false)
  if st_mdeim
    coeff = mdeim_solve(ad.mdeim_interpolation,reshape(b,:))
    recast_coefficient(ad.basis_time,coeff)
  else
    mdeim_solve(ad.mdeim_interpolation,b)
  end
end

function mdeim_solve(mdeim_interp::LU,b::AbstractArray)
  x = similar(b)
  # copyto!(x,mdeim_interp.P*b)
  # copyto!(x,mdeim_interp.L\x)
  # copyto!(x,mdeim_interp.U\x)
  ldiv!(mdeim_interp,x)
  x'
end

function recast_coefficient(
  basis_time::Vector{<:Array{T}},
  coeff::AbstractMatrix) where T

  bt,_ = basis_time
  Nt,Qt = size(bt)
  Qs = Int(length(coeff)/Qt)
  rcoeff = zeros(T,Nt,Qs)

  @fastmath @inbounds for qs in 1:Qs
    sorted_idx = [(i-1)*Qs+qs for i = 1:Qt]
    copyto!(view(rcoeff,:,qs),bt*coeff[sorted_idx])
  end

  rcoeff
end

function project_residual_coefficient(
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

function project_jacobian_coefficient(
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

function rb_contribution(
  ad::RBAffineDecomposition,
  coeff::Vector{PTArray{T}}) where T

  sz = map(*,size(ad.basis_space),size(coeff))
  rb_contrib = zeros(T,sz...)
  Threads.@threads for i = eachindex(coeff)
    LinearAlgebra.kron!(rb_contrib,ad.basis_space[i],coeff[i])
  end
  rb_contrib
end

function recast(rbspace::RBSpace,xrb::PTArray{T}) where T
  basis_space = get_basis_space(rbspace)
  basis_time = get_basis_time(rbspace)
  ns_rb = size(basis_space,2)
  nt_rb = size(basis_time,2)

  n = length(xrb)
  array = Vector{T}(undef,n)
  @inbounds for i = 1:n
    xrb_mat_i = reshape(xrb[i],nt_rb,ns_rb)
    x_i = basis_space*(basis_time*xrb_mat_i)'
    array[i] = copy(x_i)
  end

  PTArray(array)
end
