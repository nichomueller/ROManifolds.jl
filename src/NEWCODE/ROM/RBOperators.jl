function reduce_fe_operator(
  info::RBInfo,
  feop::ParamTransientFEOperator{Top},
  fesolver::ODESolver) where Top

  ϵ = info.ϵ
  # fun_mdeim = info.fun_mdeim
  nsnaps = info.nsnaps
  params = realization(feop,nsnaps)
  sols = generate_solutions(feop,fesolver,params)
  rbspace = compress_solutions(feop,fesolver,sols,params;ϵ)

  nsnaps = info.nsnaps_mdeim
  #compress_residual_and_jacobian(...)
  rb_res_c = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
  rb_jac_c = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
  rb_res = collect_residual_contributions(feop,fesolver,rbspace,rb_res_c;st_mdeim)
  rb_jac = collect_jacobian_contributions(feop,fesolver,rbspace,rb_jac_c;st_mdeim)
  rbop = RBOperator{Top}(rb_jac,rb_res,rbspace;st_mdeim)
  save(info,rbop)

  return rbop
end

struct RBOperator{Top<:OperatorType}
  res::typeof(residual_contribution)
  jac::typeof(jacobian_contribution)
  rbspace::Any
end

function residual_coefficient end

function residual_contribution end

function jacobian_coefficient end

function jacobian_contribution end

for (Top,Tslv,Trb,Tad) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver),
  (:RBSpace,:TransientRBSpace),
  (:RBAffineDecomposition,:TransientRBAffineDecomposition))

  @eval begin
    function collect_residual_contributions(
      feop::$Top,
      fesolver::$Tslv,
      a::RBAlgebraicContribution,
      rbspace::$Trb;
      kwargs...)

      meas = get_measures(a)

      res_contribs = map(get_domains(a)) do trian
        atrian = a[trian]
        nfields = length(atrian)
        Tr = typeof(residual_contribution)
        r = Vector{Tr}(undef,nfields)
        for row = 1:nfields
          ad = atrian[row]
          coeff = residual_coefficient(feop,fesolver,ad,(row,1),trian,meas;kwargs...)
          r[row] = residual_contribution(ad,coeff)
        end
        r
      end

      (args...) -> @distributed (+) for rc in res_contribs
        rc(args...)
      end
    end

    function residual_coefficient(
      feop::$Top,
      fesolver::$Tslv,
      res_ad::$Tad,
      args...;
      kwargs...)

      function _r_coefficient(u,μ)
        input = u,μ
        residual_coefficient(feop,fesolver,res_ad,input,args...;kwargs...)
      end
    end

    function residual_contribution(a::$Tad,coeff::typeof(residual_coefficient))
      function _r_contribution(u,μ)
        input = u,μ
        residual_contribution(a,coeff,input)
      end
    end

    function collect_jacobian_contributions(
      feop::$Top,
      fesolver::$Tslv,
      a::RBAlgebraicContribution,
      rbspace::$Trb;
      kwargs...)

      meas = get_measures(a)

      jac_contribs = map(get_domains(a)) do trian
        atrian = a[trian]
        nfields = length(atrian)
        Tj = typeof(jacobian_contribution)
        j = Matrix{Tj}(undef,nfields,nfields)
        for row = 1:nfields
          for col = 1:nfields
            ad = atrian[row]
            coeff = jacobian_coefficient(feop,fesolver,ad,(row,col),trian,meas;kwargs...)
            j[row,col] = jacobian_contribution(ad,coeff,(rbspace[row],rbspace[col]))
          end
        end
        j
      end

      (args...) -> @distributed (+) for jc in jac_contribs
        jc(args...)
      end
    end

    function jacobian_coefficient(
      feop::$Top,
      fesolver::$Tslv,
      res_ad::$Tad,
      args...;
      kwargs...)

      function _j_coefficient(u,μ)
        input = u,μ
        jacobian_coefficient(feop,fesolver,res_ad,input,args...;kwargs...)
      end
    end

    function jacobian_contribution(a::$Tad,coeff::typeof(jacobian_coefficient))
      function _j_contribution(u,μ)
        input = u,μ
        jacobian_contribution(a,coeff,input)
      end
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
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  meas...)

  μ,_ = input
  idx_space = res_ad.integration_domain.idx_space

  vecdata = _vecdata_residual(feop,fesolver,input,filter,trian,meas...)(μ)
  r = allocate_vector(feop.assem,vecdata)
  numeric_loop_vector!(v,feop.assem,vecdata[idx_space])

  r
end

function assemble_residual(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  res_ad::TransientRBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  meas...)

  μ,_ = input
  idx_space = res_ad.integration_domain.idx_space
  times = res_ad.integration_domain.times
  t0 = first(times)

  vecdata(t) = _vecdata_residual(feop,fesolver,input,filter,trian,meas...)(μ,t)
  r0 = allocate_vector(feop.assem,vecdata(t0))
  r = map(times) do t
    numeric_loop_vector!(r0,feop.assem,vecdata(t))
    r0[idx_space]
  end

  hcat(r...)
end

function assemble_jacobian(
  feop::ParamFEOperator,
  fesolver::FESolver,
  jac_ad::RBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  meas...)

  μ,_ = input
  idx_space = jac_ad.integration_domain.idx_space

  matdata = _matdata_jacobian(feop,fesolver,input,filter,trian,meas...)(μ)
  j = allocate_jacobian(feop.assem,matdata)
  numeric_loop_matrix!(j,feop.assem,matdata)

  reshape(j,:)[idx_space]
end

function assemble_jacobian(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  jac_ad::TransientRBAffineDecomposition,
  input::Tuple{Vararg{AbstractArray}},
  trian::Triangulation,
  filter::Tuple{Vararg{Int}},
  meas...)

  μ,_ = input
  idx_space = jac_ad.integration_domain.idx_space
  times = jac_ad.integration_domain.times
  t0 = first(times)

  matdata(t) = _matdata_jacobian(feop,fesolver,input,filter,trian,meas...)(μ,t)
  j0 = allocate_matrix(feop.assem,matdata(t0))
  j = map(times) do t
    numeric_loop_matrix!(j0,feop.assem,matdata(t))
    reshape(j0,:)[idx_space]
  end

  hcat(j...)
end

function solve(ad::RBAffineDecompositions,b::AbstractArray;st_mdeim=false)
  if st_mdeim
    coeff = solve(ad.mdeim_interpolation,reshape(b,:))
    recast_coefficient(ad,coeff)
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

function project_residual_coefficient(
  ::ODESolver,
  basis_time::Tuple{Vararg{AbstractMatrix}},
  coeff::AbstractMatrix)

  bt,btbt,btbt_shift = basis_time
  proj = Matrix{Float}[]
  @inbounds for q = axes(coeff,2), ijt = axes(bt,2)
    push!(proj,sum(bt[:,ijt].*coeff[:,q]))
  end
  proj
end

function project_jacobian_coefficient(
  ::θMethod,
  basis_time::NTuple{3,AbstractMatrix},
  coeff::AbstractMatrix)

  bt,btbt,btbt_shift = basis_time
  projs = Matrix{Float}[]
  @inbounds for q = axes(coeff,2), ijt = axes(btbt,2)
    proj = sum(btbt[:,ijt].*coeff[:,q])
    proj_shift = sum(btbt_shift[:,ijt].*coeff[2:end,q])
    push!(projs,proj+proj_shift)
  end
  projs
end

function recast_coefficient(
  basis_time::Tuple{Vararg{AbstractMatrix}},
  coeff::AbstractMatrix)

  bt,btbt,btbt_shift = basis_time
  Qs = length(coeff)
  Qt = size(basis_time,2)

  rcoeff = Matrix{Float}[]
  @inbounds for qs = 1:Qs
    sorted_idx = [(i-1)*Qs+qs for i = 1:Qt]
    push!(rcoeff,bt*coeff[sorted_idx])
  end

  rcoeff
end

for Tad in (:RBAffineDecomposition,:TransientRBAffineDecomposition)
  @eval begin
    function residual_contribution(
      res_ad::$Tad,
      coeff::typeof(residual_coefficient),
      input::Tuple{Vararg{AbstractArray}})

      bres = res_ad.basis_space
      cres = coeff(input...)
      rrb = @distributed (+) for q = axes(bres,2)
        bq = reshape(bres[:,q],nsrow,nscol)
        cq = reshape(cres[:,q],ntrow,ntcol)
        LinearAlgebra.kron(bq,cq)
      end
      rrb
    end

    function jacobian_contribution(
      jac_ad::$Tad,
      coeff::typeof(jacobian_coefficient),
      input::Tuple{Vararg{AbstractArray}})

      bjac = jac_ad.basis_space
      cjac = coeff(input...)
      jrb = @distributed (+) for q = axes(bjac,2)
        bq = reshape(bjac[:,q],nsrow,nscol)
        cq = reshape(cjac[:,q],ntrow,ntcol)
        LinearAlgebra.kron(bq,cq)
      end
      jrb
    end
  end
end
