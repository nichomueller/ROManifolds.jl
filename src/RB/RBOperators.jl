struct TransientRBOperator{Top}
  rhs::Function
  lhs::Function
  rbspace::RBSpace
end

function reduce_fe_operator(
  info::RBInfo,
  feop::ParamTransientFEOperator{Top},
  fesolver::ODESolver) where Top

  nsnaps = info.nsnaps_state
  params = realization(feop,nsnaps)
  sols = collect_solutions(feop,fesolver,params)
  rbspace = compress_snapshots(info,sols,feop,fesolver,params)
  save(info,(sols,params))

  nsnaps = info.nsnaps_system
  rb_res = collect_compress_residual(info,feop,fesolver,rbspace,sols,params)
  rb_jac = collect_compress_jacobians(info,feop,fesolver,rbspace,sols,params)

  online_res = collect_residual_contributions(info,feop,fesolver,rb_res)
  online_jac = collect_jacobians_contributions(info,feop,fesolver,rb_jac)
  rbop = TransientRBOperator{Top}(online_res,online_jac,rbspace)
  save(info,rbop)

  return rbop
end

function collect_residual_contributions(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rb_res::RBAlgebraicContribution)

  order = get_order(feop.test)
  measures = get_measures(rb_res,2*order)
  st_mdeim = info.st_mdeim

  function online_residual(online_input...)
    rb_res_contribs = []
    for trian in get_domains(rb_res)
      rb_res_trian = rb_res[trian]
      coeff = residual_coefficient(feop,fesolver,rb_res_trian,measures,online_input...;st_mdeim)
      push!(rb_res_contribs,rb_contribution(rb_res_trian,coeff))
    end
    sum(rb_res_contribs)
  end

  online_residual
end

function collect_jacobian_contributions(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rb_jacs::Vector{RBAlgebraicContribution})

  order = get_order(feop.test)
  st_mdeim = info.st_mdeim

  function online_jacobian(online_input...)
    rb_jacs_contribs = map(enumerate(rb_jacs)) do (i,rb_jac)
      measures = get_measures(rb_jac,2*order)
      rb_jac_contribs = []
      for trian in get_domains(rb_jac)
        rb_jac_trian = rb_jac[trian]
        coeff = jacobian_coefficient(feop,fesolver,rb_jac_trian,measures,online_input...;st_mdeim,i)
        push!(rb_jac_contribs,rb_contribution(rb_jac_trian,coeff))
      end
      sum(rb_jac_contribs)
    end
    sum(rb_jacs_contribs)
  end

  online_jacobian
end

function residual_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  res_ad::RBAffineDecomposition,
  args...;
  kwargs...)

  red_integr_res = assemble_residual(feop,fesolver,res_ad,args...)
  coeff = mdeim_solve(res_ad,red_integr_res;kwargs...)
  project_residual_coefficient(res_ad.basis_time,coeff)
end

function residual_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  res_ad::AbstractArray,
  args...;
  kwargs...)

  pcoeff = map(x->residual_coefficient(feop,fesolver,x,args...;kwargs...),res_ad)
  return mortar(pcoeff)
end

function jacobian_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  jac_ad::RBAffineDecomposition,
  args...;
  i::Int=1,kwargs...)

  red_integr_jac = assemble_jacobian(feop,fesolver,jac_ad,args...;i)
  coeff = mdeim_solve(jac_ad,red_integr_jac;kwargs...)
  project_jacobian_coefficient(jac_ad.basis_time,coeff)
end

function assemble_residual(
  feop::ParamTransientFEOperator,
  fesolver::θMethod,
  res_ad::RBAffineDecomposition,
  measures::Vector{Measure},
  input...)

  idx = res_ad.integration_domain.idx
  meas = res_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)

  collector = CollectResidualsMap(fesolver,feop,trian,new_meas...)
  res = collector.f(input...)
  res_cat = reduce(hcat,res)
  res_cat[idx,:]
end

function assemble_jacobian(
  feop::ParamTransientFEOperator,
  fesolver::θMethod,
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
  copyto!(x,mdeim_interp.P*b)
  copyto!(x,mdeim_interp.L\x)
  copyto!(x,mdeim_interp.U\x)
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

function rb_contribution(ad::RBAffineDecomposition,coeff::Vector{<:AbstractMatrix})
  contribs = map(LinearAlgebra.kron,ad.basis_space,coeff)
  sum(contribs)
end

function recast(rbop::TransientRBOperator,xrb::AbstractArray)
  bs = get_basis_space(rbop.rbspace)
  bt = get_basis_time(rbop.rbspace)
  rb_space_ndofs = get_rb_space_ndofs(rbop.rbspace)
  rb_time_ndofs = get_rb_time_ndofs(rbop.rbspace)
  xrb_mat = reshape(xrb,rb_time_ndofs,rb_space_ndofs)
  return bs*(bt*xrb_mat)'
end

function save(info::RBInfo,op::TransientRBOperator)
  path = joinpath(info.rb_path,"rboperator")
  save(path,op)
end

function load(T::Type{TransientRBOperator},info::RBInfo)
  path = joinpath(info.rb_path,"rboperator")
  load(T,path)
end
