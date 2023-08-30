struct TransientRBOperator{Top}
  res::Function
  jac::Function
  rbspace::TransientRBSpace
end

function reduce_fe_operator(
  info::RBInfo,
  feop::ParamTransientFEOperator{Top},
  fesolver::ODESolver;
  st_mdeim=true) where Top

  ϵ = info.ϵ
  # fun_mdeim = info.fun_mdeim
  nsnaps = info.nsnaps_state
  params = realization(feop,nsnaps)
  sols = collect_solutions(feop,fesolver,params)
  rbspace = compress_solutions(feop,fesolver,sols,params;ϵ)

  nsnaps = info.nsnaps_system
  rb_jac = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps,st_mdeim)
  rb_jac = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps,st_mdeim)

  rb_res = compress_residuals(feop,fesolver,rbspace,snaps,params;ϵ,nsnaps,st_mdeim)
  rbop = TransientRBOperator{Top}(rb_res,rb_jac,rbspace)
  save(info,rbop)

  return rbop
end

function collect_residual_contributions(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rbspace::SingleFieldRBSpace,
  rb_res::RBAlgebraicContribution;
  kwargs...)

  order = get_order(feop.test)
  measures = get_measures(rb_res,2*order)

  function online_residual(online_input...)
    rb_res_contribs = map(get_domains(rb_res)) do trian
      rb_res_trian = rb_res[trian]
      coeff = residual_coefficient(feop,fesolver,rb_res_trian,measures,online_input...;kwargs...)
      rb_contribution(rb_res_trian,coeff)
    end
    sum(rb_res_contribs)
  end

  online_residual
end

function collect_residual_contributions(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rbspace::MultiFieldRBSpace,
  rb_res::RBAlgebraicContribution;
  kwargs...)

  order = get_order(feop.test)
  measures = get_measures(rb_res,2*order)
  nfields = get_nfields(rbspace)

  function online_residual(online_input...)
    rb_res_contribs = map(get_domains(rb_res)) do trian
      rb_res_trian = rb_res[trian]
      all_idx = index_pairs(1:nfields,1)

      rb_res_trian_contribs = map(all_idx) do filter
        filt_op = filter_operator(feop,filter)
        coeff = residual_coefficient(filt_op,fesolver,rb_res_trian,measures,online_input...;kwargs...)
        rb_contribution(rb_res_trian,coeff)
      end
      mortar(rb_res_trian_contribs)
    end

    sum(rb_res_contribs)
  end

  online_residual
end

function residual_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  res_ad::RBAffineDecomposition,
  args...;
  kwargs...)

  red_integr_res = assemble_residual(feop,fesolver,res_ad,args...)
  coeff = mdeim_solve(res_ad,red_integr_res;kwargs...)
  project_residual_coefficient(fesolver,res_ad.basis_time,coeff)
end

function jacobian_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  jac_ad::RBAffineDecomposition,
  args...;
  i::Int=1,kwargs...)

  red_integr_jac = assemble_jacobian(feop,fesolver,jac_ad,args...;i)
  coeff = mdeim_solve(jac_ad,red_integr_jac;kwargs...)
  project_jacobian_coefficient(fesolver,jac_ad.basis_time,coeff)
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
  input...)

  idx = jac_ad.integration_domain.idx
  meas = jac_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)

  collector = CollectJacobiansMap(fesolver,feop,trian,new_meas...)
  jac = collector.f(input...)
  jac_cat = reduce(hcat,jac)
  jac_cat.nonzero_val[idx]
end

function mdeim_solve(ad::RBAffineDecomposition,b::AbstractArray;st_mdeim=true)
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
  basis_time::Tuple{Vararg{AbstractArray}},
  coeff::AbstractMatrix)

  bt,_ = basis_time
  Qt = size(bt,2)
  Qs = Int(length(coeff)/Qt)

  rcoeff = map(1:Qs) do qs
    sorted_idx = [(i-1)*Qs+qs for i = 1:Qt]
    bt*coeff[sorted_idx]
  end

  hcat(rcoeff...)
end

function project_residual_coefficient(
  ::ODESolver,
  basis_time::Tuple{Vararg{AbstractArray}},
  coeff::AbstractMatrix)

  _,bt_proj = basis_time
  proj = map(eachcol(coeff)) do c
    pc = map(eachcol(bt_proj)) do b
      sum(b.*c)
    end
    reshape(pc,:,1)
  end
  proj
end

function project_jacobian_coefficient(
  ::θMethod,
  basis_time::Tuple{Vararg{AbstractArray}},
  coeff::AbstractMatrix)

  _,bt_proj = basis_time
  proj = map(eachcol(coeff)) do c
    pcr = map(axes(bt_proj,3)) do col
      map(axes(bt_proj,2)) do row
        sum(bt_proj[:,row,col].*c)
      end
    end
    hcat(pcr...)
  end
  proj
end

function rb_contribution(ad::RBAffineDecomposition,coeff::Vector{<:AbstractMatrix})
  contribs = lazy_map(LinearAlgebra.kron,ad.basis_space,coeff)
  sum(contribs)
end

function recast(rbop::TransientRBOperator,xrb::AbstractArray)
  bs = get_basis_space(rbop.rbspace)
  bt = get_basis_time(rbop.rbspace)
  rb_space_ndofs = get_rb_space_ndofs(rbop.rbspace)
  rb_time_ndofs = get_rb_time_ndofs(rbop.rbspace)
  xrb_mat = reshape(xrb,rb_time_ndofs,rb_space_ndofs)
  bs*(bt*xrb_mat)'
end

function save(info::RBInfo,op::TransientRBOperator)
  path = joinpath(info.rb_path,"rboperator")
  save(path,op)
end

function load(T::Type{TransientRBOperator},info::RBInfo)
  path = joinpath(info.rb_path,"rboperator")
  load(T,path)
end
