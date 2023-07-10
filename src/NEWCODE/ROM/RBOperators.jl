struct RBOperator{Top<:OperatorType}
  res::ParamArray
  jac::ParamArray
  rbspace::Any
end

for (Top,Tslv,Tad) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver),
  (:RBAffineDecomposition,:TransientRBAffineDecomposition))

  @eval begin
    function reduce_fe_operator(
      info::RBInfo,
      feop::$Top{T},
      fesolver::$Tslv) where T

      ϵ = info.ϵ
      # fun_mdeim = info.fun_mdeim
      nsnaps = info.nsnaps_state
      params = realization(feop,nsnaps)
      sols = generate_solutions(feop,fesolver,params)
      rbspace = compress_solutions(feop,fesolver,sols,params;ϵ)

      nsnaps = info.nsnaps_system
      #compress_residual_and_jacobian(...)
      rb_res_c = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
      rb_jac_c = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
      rb_res = collect_residual_contributions(feop,fesolver,rb_res_c;st_mdeim)
      rb_jac = collect_jacobian_contributions(feop,fesolver,rb_jac_c;st_mdeim)
      rbop = RBOperator{T}(rb_jac,rb_res,rbspace;st_mdeim)
      save(info,rbop)

      return rbop
    end

    function collect_residual_contributions(
      feop::$Top,
      fesolver::$Tslv,
      a::RBAlgebraicContribution;
      kwargs...)

      measures = get_measures(a)
      nfields = get_nfields(a)
      r = Vector{ParamArray}(undef,nfields)

      for (m,ad) in a.dict
        for row = 1:nfields
          ad_r = ad[row]
            rr = residual_contribution(
            feop,
            fesolver,
            ad_r,
            (row,1),
            measures;
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

      function _r_contribution(u,μ)
        input = u,μ
        coeff = residual_coefficient(feop,fesolver,res_ad,input,args...;kwargs...)
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

      function _r_contribution(u,μ)
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
      nfields = get_nfields(a)
      j = Matrix{ParamArray}(undef,nfields,nfields)

      for (_,ad) in a.dict
        for row = 1:nfields, col = 1:nfields
          ad_rc = ad[row,col]
          jrc = jacobian_contribution(
            feop,
            fesolver,
            ad_rc,
            (row,col),
            measures;
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

      function _j_contribution(u,μ)
        input = u,μ
        coeff = jacobian_coefficient(feop,fesolver,jac_ad,input,args...;kwargs...)
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

      function _j_contribution(u,μ)
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
  input::Tuple{Vararg{AbstractArray}},
  args...)

  red_integr_res = assemble_residual(feop,fesolver,res_ad,input,args...)
  solve(res_ad,red_integr_res)
end

function residual_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  res_ad::TransientRBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  args...;
  kwargs...)

  red_integr_res = assemble_residual(feop,fesolver,res_ad,input,args...)
  coeff = solve(res_ad,red_integr_res;kwargs...)
  project_residual_coefficient(fesolver,res_ad.basis_time,coeff)
end

function residual_jacobian(
  feop::ParamFEOperator,
  fesolver::FESolver,
  jac_ad::RBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  args...)

  jac_integr_res = assemble_jacobian(feop,fesolver,jac_ad,input,args...)
  solve(jac_ad,jac_integr_res)
end

function jacobian_coefficient(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  jac_ad::TransientRBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  args...;
  kwargs...)

  red_integr_jac = assemble_jacobian(feop,fesolver,jac_ad,input,args...)
  coeff = solve(jac_ad,red_integr_jac;kwargs...)
  project_jacobian_coefficient(fesolver,jac_ad.basis_time,coeff)
end

function assemble_residual(
  feop::ParamFEOperator,
  fesolver::FESolver,
  res_ad::RBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  filter::Tuple{Vararg{Int}},
  measures::Vector{Measure})

  u,μ = input
  idx = res_ad.integration_domain.idx
  meas = res_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)

  vecdata = _vecdata_residual(feop,fesolver,u,μ,filter,new_meas...;trian)
  r = allocate_vector(feop.assem,vecdata)
  numeric_loop_vector!(v,feop.assem,vecdata[idx])

  r
end

function assemble_residual(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  res_ad::TransientRBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  filter::Tuple{Vararg{Int}},
  measures::Vector{Measure})

  u,μ = input
  idx = res_ad.integration_domain.idx
  meas = res_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)
  times = res_ad.integration_domain.times
  t0 = first(times)

  vecdata = _vecdata_residual(feop,fesolver,u,μ,filter,new_meas...;trian)
  r0 = allocate_vector(feop.assem,vecdata(μ,t0))
  r = map(times) do t
    numeric_loop_vector!(r0,feop.assem,vecdata(μ,t))
    r0[idx]
  end

  hcat(r...)
end

function assemble_jacobian(
  feop::ParamFEOperator,
  fesolver::FESolver,
  jac_ad::RBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  filter::Tuple{Vararg{Int}},
  measures::Vector{Measure})

  u,μ = input
  idx = jac_ad.integration_domain.idx
  meas = jac_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)

  matdata = _matdata_jacobian(feop,fesolver,u,μ,filter,new_meas...;trian)
  j = allocate_jacobian(feop.assem,matdata)
  numeric_loop_matrix!(j,feop.assem,matdata)

  Vector(reshape(j,:)[idx])
end

function assemble_jacobian(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  jac_ad::TransientRBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  filter::Tuple{Vararg{Int}},
  measures::Vector{Measure})

  u,μ = input
  idx = jac_ad.integration_domain.idx
  meas = jac_ad.integration_domain.meas
  trian = get_triangulation(meas)
  new_meas = modify_measures(measures,meas)
  times = jac_ad.integration_domain.times
  t0 = first(times)

  matdata = _matdata_jacobian(feop,fesolver,u,μ,filter,new_meas...;trian)
  j0 = allocate_matrix(feop.assem,matdata(μ,t0))
  j = map(times) do t
    numeric_loop_matrix!(j0,feop.assem,matdata(μ,t))
    Vector(reshape(j0,:)[idx])
  end

  hcat(j...)
end

function solve(ad::RBAffineDecompositions,b::AbstractArray;st_mdeim=true)
  if st_mdeim
    coeff = solve(ad.mdeim_interpolation,reshape(b,:))
    recast_coefficient(ad.basis_time,coeff)
  else
    solve(ad.mdeim_interpolation,b)
  end
end

function solve(mdeim_interp::LU,b::AbstractArray)
  ns = LUNumericalSetup(lu(mdeim_interp))
  x = similar(b)
  solve!(x,ns,b)
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

  _,btbt = basis_time
  proj = map(eachcol(coeff)) do c
    pc = map(eachcol(btbt)) do b
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

  _,btbt = basis_time
  proj = map(eachcol(coeff)) do c
    pcr = map(axes(btbt,3)) do col
      map(axes(btbt,2)) do row
        sum(btbt[:,row,col].*c)
      end
    end
    hcat(pcr...)
  end
  proj
end
