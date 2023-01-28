function steady_poisson_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lhs = eval_on_structure(rbpos,:A,μ)
  rhs = eval_on_structure(rbpos,(:F,:H,:A_lift),μ)
  lhs,sum(rhs)
end

function unsteady_poisson_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lhs = eval_on_structure(rbpos,(:A,:M),μ)
  rhs = eval_on_structure(rbpos,(:F,:H,:A_lift,:M_lift),μ)
  sum(lhs),sum(rhs)
end

function steady_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lhs = eval_on_structure(rbpos,(:A,:B),μ)
  rhs = eval_on_structure(rbpos,(:F,:H,:A_lift,:B_lift),μ)

  np = size(lhs[2],1)
  rb_lhs = vcat(hcat(lhs[1],-lhs[2]'),hcat(lhs[2],zeros(np,np)))
  rb_rhs = vcat(sum(rhs[1:end-1]),rhs[end])
  rb_lhs,rb_rhs
end

function unsteady_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lhs = eval_on_structure(rbpos,(:A,:B,:BT,:M),μ)
  rhs = eval_on_structure(rbpos,(:F,:H,:A_lift,:B_lift,:M_lift),μ)

  np = size(lhs[2],1)
  rb_lhs = vcat(hcat(lhs[1]+lhs[4],-lhs[3]),hcat(lhs[2],zeros(np,np)))
  rb_rhs = vcat(rhs[1]+rhs[2]+rhs[3]+rhs[5],rhs[4])
  rb_lhs,rb_rhs
end

function steady_navier_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lin_rb_lhs,lin_rb_rhs = steady_stokes_rb_system(rbpos,μ)
  nonlin_lhs(u) = eval_on_structure(rbpos,(:C,:D),u)
  nonlin_rhs(u) = eval_on_structure(rbpos,:C_lift,u)

  opA,opB = get_op(rbpos,(:A,:B))
  rbu,rbp = get_rbspace_row(opA),get_rbspace_row(opB)
  nu,np = get_ns(rbu),get_ns(rbp)

  block12,block21,block22 = zeros(nu,np),zeros(np,nu),zeros(np,np)
  nonlin_rb_lhs1(u) = vcat(hcat(nonlin_lhs[1](u),block12),
                           hcat(block21,block22))
  nonlin_rb_lhs2(u) = vcat(hcat(nonlin_lhs[1](u)+nonlin_lhs[2](u),block12),
                           hcat(block21,block22))
  nonlin_rb_rhs(ud) = vcat(nonlin_rhs(ud),zeros(np,1))

  jac_rb(u) = lin_rb_lhs + nonlin_rb_lhs2(u)
  lhs_rb(u) = lin_rb_lhs + nonlin_rb_lhs1(u)
  rhs_rb(ud) = lin_rb_rhs + nonlin_rb_rhs(ud)
  res_rb(u,ud,x_rb) = lhs_rb(u)*x_rb - rhs_rb(ud)

  res_rb,jac_rb
end

function unsteady_navier_stokes_rb_system(
  rbpos::NTuple{N,RBParamOnlineStructure},
  μ::Param) where N

  lin_rb_lhs,lin_rb_rhs = unsteady_stokes_rb_system(rbpos,μ)
  nonlin_lhs(uθ) = eval_on_structure(rbpos,(:C,:D),μ,uθ)
  nonlin_rhs(uθ) = eval_on_structure(rbpos,:C_lift,μ,uθ)

  opA,opB = get_op(rbpos,(:A,:B))
  rbu,rbp = get_rbspace_row(opA),get_rbspace_row(opB)
  nu,np = get_ns(rbu)*get_nt(rbu),get_ns(rbp)*get_nt(rbp)

  block12,block21,block22 = zeros(nu,np),zeros(np,nu),zeros(np,np)

  function nonlin_rb_lhs1(uθ)
    Cuθ,_ = nonlin_lhs(uθ)
    vcat(hcat(Cuθ,block12),hcat(block21,block22))
  end

  function nonlin_rb_lhs2(uθ)
    Cuθ,Duθ = nonlin_lhs(uθ)
    vcat(hcat(Cuθ+Duθ,block12),hcat(block21,block22))
  end

  nonlin_rb_rhs(uθ) = vcat(nonlin_rhs(uθ),zeros(np,1))

  jac_rb(uθ) = lin_rb_lhs + nonlin_rb_lhs2(uθ)
  lhs_rb(uθ) = lin_rb_lhs + nonlin_rb_lhs1(uθ)
  rhs_rb(uθ) = lin_rb_rhs + nonlin_rb_rhs(uθ)
  res_rb(uθ,x_rb) = lhs_rb(uθ)*x_rb - rhs_rb(uθ)

  res_rb,jac_rb
end

function solve_rb_system(rb_lhs::Matrix{Float},rb_rhs::Matrix{Float})
  rb_lhs \ rb_rhs
end

function solve_rb_system(
  res::Function,
  jac::Function,
  x0::Matrix{Float},
  fespaces::NTuple{2,FESpace},
  rbspace::NTuple{2,RBSpace};
  tol=1e-10,maxtol=1e10,maxit=10)

  Uk,Vk = fespaces
  bsu = get_basis_space(rbspace[1])
  nsu = size(bsu,2)
  x_rb = x0

  u(x_rb::AbstractArray) = FEFunction(Vk,bsu*x_rb[1:nsu])
  ud(x_rb::AbstractArray) = FEFunction(Uk,bsu*x_rb[1:nsu])

  err = 1.
  iter = 0
  while norm(err) > tol && iter < maxit
    if norm(err) > maxtol
      println("Newton iterations did not converge")
      return x_rb
    end
    jx_rb,rx_rb = jac(u(x_rb)),res(u(x_rb),ud(x_rb),x_rb)
    err = jx_rb \ rx_rb
    x_rb -= err
    iter += 1
    println("Newton method: err = $(norm(err)), iter = $iter")
  end

  x_rb
end

function solve_rb_system(
  res::Function,
  jac::Function,
  x0::Matrix{Float},
  Uk::Function,
  rbspace::NTuple{2,<:RBSpace},
  time_info::TimeInfo,
  tol=1e-10,maxtol=1e10,maxit=10)

  timesθ = get_timesθ(time_info)
  θ = get_θ(time_info)
  Ns = get_Ns(rbspace[1])
  nstu = get_ns(rbspace[1])*get_nt(rbspace[1])

  function get_uθ_fun(ufe::Matrix{Float})
    uθfe = compute_in_timesθ(ufe,θ)
    n(tθ) = findall(x -> x == tθ,timesθ)[1]
    tθ -> FEFunction(Uk(tθ),uθfe[:,n(tθ)])
  end

  function get_uθ_fun(x_rb::Vector{Float})
    ufe = reconstruct_fe_sol(rbspace[1],x_rb[1:nstu])
    uθfe = compute_in_timesθ(ufe,θ)
    n(tθ) = findall(x -> x == tθ,timesθ)[1]
    tθ -> FEFunction(Uk(tθ),uθfe[:,n(tθ)])
  end

  uθ_fun = get_uθ_fun(x0)
  u0_rb = rb_spacetime_projection(rbspace[1],x0[1:Ns,:])
  p0_rb = rb_spacetime_projection(rbspace[2],x0[Ns+1:end,:])
  x_rb = vcat(u0_rb,p0_rb)[:,1]

  err = 1.
  iter = 0
  while norm(err) > tol && iter < maxit

    if norm(err) > maxtol
      println("Newton iterations did not converge")
      return x_rb
    end

    jx_rb,rx_rb = jac(uθ_fun),res(uθ_fun,x_rb)
    err = jx_rb \ rx_rb
    x_rb -= err[:,1]
    uθ_fun = get_uθ_fun(x_rb)
    iter += 1

    println("Newton method: err = $(norm(err)), iter = $iter")
  end

  x_rb
end

function initial_guess(
  uh::Snapshots,
  ph::Snapshots,
  μ::Vector{Param},
  μk::Param)

  kmin = nearest_parameter(μ,μk)
  vcat(get_snap(uh[kmin]),get_snap(ph[kmin]))
end

function nearest_parameter(μ::Vector{Param},μk::Param)
  vars = [var(μi-μk) for μi=μ]
  argmin(vars)
end

function reconstruct_fe_sol(rbspace::RBSpaceSteady,rb_sol::Array{Float})
  bs = get_basis_space(rbspace)
  bs*rb_sol
end

function reconstruct_fe_sol(rbspace::RBSpaceUnsteady,rb_sol::Array{Float})
  bs = get_basis_space(rbspace)
  bt = get_basis_time(rbspace)
  ns = get_ns(rbspace)
  nt = get_nt(rbspace)

  rb_sol_resh = reshape(rb_sol,nt,ns)
  bs*(bt*rb_sol_resh)'
end

function reconstruct_fe_sol(rbspace::NTuple{2,RBSpaceSteady},rb_sol::Array{Float})
  bs_u,bs_p = get_basis_space.(rbspace)

  ns = get_ns.(rbspace)
  rb_sol_u,rb_sol_p = rb_sol[1:ns[1],:],rb_sol[1+ns[1]:end,:]

  bs_u*rb_sol_u,bs_p*rb_sol_p
end

function reconstruct_fe_sol(rbspace::NTuple{2,RBSpaceUnsteady},rb_sol::Array{Float})
  bs_u,bs_p = get_basis_space.(rbspace)
  bt_u,bt_p = get_basis_time.(rbspace)

  ns = get_ns.(rbspace)
  nt = get_nt.(rbspace)
  n = ns.*nt
  rb_sol_u,rb_sol_p = rb_sol[1:n[1],:],rb_sol[1+n[1]:end,:]
  u_rb_resh = reshape(rb_sol_u,nt[1],ns[1])
  p_rb_resh = reshape(rb_sol_p,nt[2],ns[2])

  bs_u*(bt_u*u_rb_resh)',bs_p*(bt_p*p_rb_resh)'
end
