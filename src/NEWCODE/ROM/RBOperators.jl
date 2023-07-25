abstract type GenericRBOperator{Top<:OperatorType} end

struct RBOperator{Top} <: GenericRBOperator{Top}
  res::Vector{ParamArray}
  jac::Matrix{ParamArray}
  rbspace::RBSpace
end

function reduce_fe_operator(
  info::RBInfo,
  feop::ParamFEOperator{Top},
  fesolver::FESolver) where Top

  ϵ = info.ϵ
  # fun_mdeim = info.fun_mdeim
  nsnaps = info.nsnaps_state
  params = realization(feop,nsnaps)
  sols = collect_solutions(feop,fesolver,params)
  rbspace = compress_solutions(feop,fesolver,sols,params;ϵ)

  nsnaps = info.nsnaps_system
  rb_res_c = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
  rb_jac_c = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
  rb_res = collect_residual_contributions(feop,fesolver,rb_res_c;st_mdeim)
  rb_jac = collect_jacobian_contributions(feop,fesolver,rb_jac_c;st_mdeim)
  rbop = RBOperator{Top}(rb_res,rb_jac,rbspace)
  save(info,rbop)

  return rbop
end

struct TransientRBOperator{Top} <: GenericRBOperator{Top}
  res::Vector{ParamArray}
  jac::Matrix{ParamArray}
  djac::Matrix{ParamArray}
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
  save(info,(sols,params))

  nsnaps = info.nsnaps_system
  rb_res_c = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps,st_mdeim)
  rb_jac_c = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps,st_mdeim)
  rb_djac_c = compress_djacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps,st_mdeim)
  rb_res = collect_residual_contributions(feop,fesolver,rb_res_c;st_mdeim)
  rb_jac = collect_jacobian_contributions(feop,fesolver,rb_jac_c;st_mdeim)
  rb_djac = collect_jacobian_contributions(feop,fesolver,rb_djac_c;i=2,st_mdeim)
  rbop = TransientRBOperator{Top}(rb_res,rb_jac,rb_djac,rbspace)
  save(info,rbop)

  return rbop
end

for (Top,Tslv,Tad) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver),
  (:RBAffineDecomposition,:TransientRBAffineDecomposition))

  @eval begin

    function collect_residual_contributions(
      feop::$Top,
      fesolver::$Tslv,
      a::RBAlgebraicContribution;
      kwargs...)

      measures = get_measures(a)
      nfields = num_fields(a)
      r = Vector{ParamArray}(undef,nfields)

      for (m,ad) in a.dict
        for row = 1:nfields
          ad_r = ad[row]
          rr = residual_contribution(
            feop,
            fesolver,
            ad_r,
            measures,
            (row,1);
            kwargs...)
          add_contribution!(r,rr,row)
        end
      end

      r
    end

    function residual_contribution(
      feop::$Top,
      fesolver::$Tslv,
      res_ad::$Tad,
      args...;
      kwargs...)

      function _r_contribution(input...)
        new_args = (args...,input...)
        coeff = residual_coefficient(feop,fesolver,res_ad,new_args...;kwargs...)
        rb_contribution(res_ad,coeff)
      end
      ParamArray(_r_contribution)
    end

    function residual_contribution(
      ::$Top,
      ::$Tslv,
      res_ad::ZeroRBAffineDecomposition,
      args...;
      kwargs...)

      function _r_contribution(input...)
        res_ad.proj
      end
      ParamArray(_r_contribution)
    end

    function collect_jacobian_contributions(
      feop::$Top,
      fesolver::$Tslv,
      a::RBAlgebraicContribution;
      kwargs...)

      measures = get_measures(a)
      nfields = num_fields(a)
      j = Matrix{ParamArray}(undef,nfields,nfields)

      for (_,ad) in a.dict
        for row = 1:nfields, col = 1:nfields
          ad_rc = ad[row,col]
          jrc = jacobian_contribution(
            feop,
            fesolver,
            ad_rc,
            measures,
            (row,col);
            kwargs...)
          add_contribution!(j,jrc,row,col)
        end
      end

      j
    end

    function jacobian_contribution(
      feop::$Top,
      fesolver::$Tslv,
      jac_ad::$Tad,
      args...;
      kwargs...)

      function _j_contribution(input...)
        new_args = (args...,input...)
        coeff = jacobian_coefficient(feop,fesolver,jac_ad,new_args...;kwargs...)
        rb_contribution(jac_ad,coeff)
      end
      ParamArray(_j_contribution)
    end

    function jacobian_contribution(
      ::$Top,
      ::$Tslv,
      jac_ad::ZeroRBAffineDecomposition,
      args...;
      kwargs...)

      function _j_contribution(input...)
        jac_ad.proj
      end
      ParamArray(_j_contribution)
    end

    function rb_contribution(
      ad::$Tad,
      coeff::Vector{<:AbstractMatrix})

      contribs = map(LinearAlgebra.kron,ad.basis_space,coeff)
      sum(contribs)
    end
  end
end

function residual_coefficient(
  feop::ParamFEOperator,
  fesolver::FESolver,
  res_ad::RBAffineDecomposition,
  args...)

  red_integr_res = assemble_residual(feop,fesolver,res_ad,args...)
  solve(res_ad,red_integr_res)
end

function residual_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  res_ad::TransientRBAffineDecomposition,
  args...;
  kwargs...)

  red_integr_res = assemble_residual(feop,fesolver,res_ad,args...)
  coeff = solve(res_ad,red_integr_res;kwargs...)
  project_residual_coefficient(fesolver,res_ad.basis_time,coeff)
end

function jacobian_coefficient(
  feop::ParamFEOperator,
  fesolver::FESolver,
  jac_ad::RBAffineDecomposition,
  args...)

  jac_integr_res = assemble_jacobian(feop,fesolver,jac_ad,args...)
  solve(jac_ad,jac_integr_res)
end

function jacobian_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  jac_ad::TransientRBAffineDecomposition,
  args...;
  i::Int=1,kwargs...)

  red_integr_jac = assemble_jacobian(feop,fesolver,jac_ad,args...;i)
  coeff = solve(jac_ad,red_integr_jac;kwargs...)
  project_jacobian_coefficient(fesolver,jac_ad.basis_time,coeff)
end

# function assemble_residual(
#   feop::ParamFEOperator,
#   fesolver::FESolver,
#   res_ad::RBAffineDecomposition,
#   measures::Vector{Measure},
#   filter::Tuple{Vararg{Int}},
#   input...)

#   μ, = input
#   idx = res_ad.integration_domain.idx
#   meas = res_ad.integration_domain.meas
#   trian = get_triangulation(meas)
#   new_meas = modify_measures(measures,meas)

#   vecdata = _vecdata_residual(feop,fesolver,u,μ,filter,new_meas...;trian)
#   r = allocate_vector(feop.assem,vecdata)
#   numeric_loop_vector!(v,feop.assem,vecdata[idx])

#   r
# end

function assemble_residual(
  feop::ParamTransientFEOperator{Affine},
  fesolver::θMethod,
  res_ad::TransientRBAffineDecomposition,
  measures::Vector{Measure},
  filter::Tuple{Vararg{Int}},
  input...)

  μ, = input
  idx = res_ad.integration_domain.idx
  meas = res_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)
  times = res_ad.integration_domain.times

  res_iter = init_vec_iterator(feop,fesolver,trian,filter,new_meas...)
  r = map(times) do t
    update!(res_iter,feop,fesolver,μ,t)
    r_t = evaluate!(res_iter)
    r_t[idx]
  end

  hcat(r...)
end

function assemble_residual(
  feop::ParamTransientFEOperator,
  fesolver::θMethod,
  res_ad::TransientRBAffineDecomposition,
  measures::Vector{Measure},
  filter::Tuple{Vararg{Int}},
  input...)

  μ,u = input
  idx = res_ad.integration_domain.idx
  meas = res_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)
  times = res_ad.integration_domain.times

  θ = solver.θ
  ic = solver.uh0(μ)
  ich = get_free_dof_values(ic)
  prev_u = hcat(ich,u[:,1:end-1])
  uθ = θ*u[:,2:end] + (1-θ)*prev_u

  res_iter = init_vec_iterator(feop,fesolver,trian,filter,new_meas...)
  r = map(times) do t
    uθt = uθ[:,nt]
    update!(res_iter,feop,fesolver,μ,t,uθt)
    r_t = evaluate!(res_iter)
    r_t[idx]
  end

  hcat(r...)
end

# function assemble_jacobian(
#   feop::ParamFEOperator,
#   fesolver::FESolver,
#   jac_ad::RBAffineDecomposition,
#   input,
#   filter::Tuple{Vararg{Int}},
#   measures::Vector{Measure})

#   u,μ = input
#   idx = jac_ad.integration_domain.idx
#   meas = jac_ad.integration_domain.meas
#   trian = get_triangulation(meas)
#   new_meas = modify_measures(measures,meas)

#   matdata = _matdata_jacobian(feop,fesolver,u,μ,filter,new_meas...;trian)
#   j = allocate_jacobian(feop.assem,matdata)
#   numeric_loop_matrix!(j,feop.assem,matdata)

#   Vector(reshape(j,:)[idx])
# end

function assemble_jacobian(
  feop::ParamTransientFEOperator{Affine},
  fesolver::θMethod,
  jac_ad::TransientRBAffineDecomposition,
  measures::Vector{Measure},
  filter::Tuple{Vararg{Int}},
  input...;
  i::Int=1)

  μ, = input
  idx = jac_ad.integration_domain.idx
  meas = jac_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)
  times = jac_ad.integration_domain.times

  jac_iter = init_mat_iterator(feop,fesolver,trian,filter,new_meas...;i)
  j = map(times) do t
    update!(jac_iter,feop,fesolver,μ,t)
    j_t = evaluate!(jac_iter)
    j_t.nonzero_val[idx]
  end

  hcat(j...)
end

function assemble_jacobian(
  feop::ParamTransientFEOperator,
  fesolver::θMethod,
  jac_ad::TransientRBAffineDecomposition,
  measures::Vector{Measure},
  filter::Tuple{Vararg{Int}},
  input...;
  i::Int=1)

  μ,u = input
  idx = jac_ad.integration_domain.idx
  meas = jac_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)
  times = jac_ad.integration_domain.times

  θ = solver.θ
  ic = solver.uh0(μ)
  ich = get_free_dof_values(ic)
  prev_u = hcat(ich,u[:,1:end-1])
  uθ = θ*u[:,2:end] + (1-θ)*prev_u

  jac_iter = init_mat_iterator(feop,fesolver,trian,filter,new_meas...;i)
  j = map(times) do t
    uθt = uθ[:,nt]
    update!(jac_iter,feop,fesolver,μ,t,uθt)
    j_t = evaluate!(jac_iter)
    j_t.nonzero_val[idx]
  end

  hcat(j...)
end

function Gridap.Algebra.solve(ad::RBAffineDecompositions,b::AbstractArray;st_mdeim=true)
  if st_mdeim
    coeff = solve(ad.mdeim_interpolation,reshape(b,:))
    recast_coefficient(ad.basis_time,coeff)
  else
    solve(ad.mdeim_interpolation,b)
  end
end

function Gridap.Algebra.solve(mdeim_interp::LU,b::AbstractArray)
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

function recast(rbop::RBOperator,xrb::AbstractArray)
  bs = get_basis_space(rbop.rbspace)
  bs*xrb
end

function recast(rbop::TransientRBOperator,xrb::AbstractArray)
  bs = get_basis_space(rbop.rbspace)
  bt = get_basis_time(rbop.rbspace)
  rb_space_ndofs = get_rb_space_ndofs(rbop.rbspace)
  rb_time_ndofs = get_rb_time_ndofs(rbop.rbspace)
  xrb_mat = reshape(xrb,rb_time_ndofs,rb_space_ndofs)
  bs*(bt*xrb_mat)'
end

function save(info::RBInfo,op::GenericRBOperator)
  path = joinpath(info.rb_path,"rboperator")
  save(path,op)
end

function load(T::Type{GenericRBOperator},info::RBInfo)
  path = joinpath(info.rb_path,"rboperator")
  load(T,path)
end
