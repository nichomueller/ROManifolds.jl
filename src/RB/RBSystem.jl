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
  nonlin_lhs(uθ_rb) = eval_on_structure(rbpos,(:C,:D),uθ_rb)
  nonlin_rhs(uθ_rb) = eval_on_structure(rbpos,:C_lift,uθ_rb)

  opA,opB = get_op(rbpos,(:A,:B))
  rbu,rbp = get_rbspace_row(opA),get_rbspace_row(opB)
  nu,np = get_ns(rbu)*get_nt(rbu),get_ns(rbp)*get_nt(rbp)

  block12,block21,block22 = zeros(nu,np),zeros(np,nu),zeros(np,np)

  function nonlin_rb_lhs1(uθ_rb)
    Cuθ_rb,_ = nonlin_lhs(uθ_rb)
    vcat(hcat(Cuθ_rb,block12),hcat(block21,block22))
  end

  function nonlin_rb_lhs2(uθ_rb)
    Cuθ_rb,Duθ_rb = nonlin_lhs(uθ_rb)
    vcat(hcat(Cuθ_rb+Duθ_rb,block12),hcat(block21,block22))
  end

  nonlin_rb_rhs(uθ_rb,uθd_rb) = vcat(nonlin_rhs(uθ_rb)*uθd_rb,zeros(np,1))

  jac_rb(uθ_rb) = lin_rb_lhs + nonlin_rb_lhs2(uθ_rb)
  lhs_rb(uθ_rb) = lin_rb_lhs + nonlin_rb_lhs1(uθ_rb)
  rhs_rb(uθ_rb,uθd_rb) = lin_rb_rhs + nonlin_rb_rhs(uθ_rb,uθd_rb)
  res_rb(uθ_rb,uθd_rb,x_rb) = lhs_rb(uθ_rb)*x_rb - rhs_rb(uθ_rb,uθd_rb)

  res_rb,jac_rb
end

function solve_rb_system(rb_lhs::Matrix{Float},rb_rhs::Matrix{Float})
  println("Solving system via backslash")
  rb_lhs \ rb_rhs
end

function solve_rb_system(
  res::Function,
  jac::Function,
  x0::Matrix{Float},
  fespaces::NTuple{2,FESpace},
  rbspace::NTuple{2,RBSpace};
  tol=1e-10,maxit=10)

  println("Solving system via Newton method")

  Uk,Vk = fespaces
  bsu = get_basis_space(rbspace[1])
  nsu = size(bsu,2)
  x_rb = x0

  u(x_rb::AbstractArray) = FEFunction(Vk,bsu*x_rb[1:nsu])
  ud(x_rb::AbstractArray) = FEFunction(Uk,bsu*x_rb[1:nsu])

  err = 1.
  iter = 0
  while norm(err) > tol && iter < maxit
    jx_rb,rx_rb = jac(u(x_rb)),res(u(x_rb),ud(x_rb),x_rb)
    err = jx_rb \ rx_rb
    x_rb -= err
    iter += 1
    println("err = $(norm(err)), iter = $iter")
  end

  x_rb
end

function solve_rb_system(
  res::Function,
  jac::Function,
  x0::Vector{Float},
  ud::Vector{Float},
  rbspace::NTuple{2,<:RBSpace},
  rbspaceθ::NTuple{2,<:RBSpace},
  timesθ::Vector{<:Real},
  θ::Real,
  tol=1e-10,maxit=10)

  println("Solving system via Newton method")

  bstu = get_basis_spacetime(rbspace[1])
  nstu = size(bstu,2)
  x_rb = x0
  bstuθ = get_basis_spacetime(rbspaceθ[1])
  bstuθd = get_basis_spacetime(rbspaceθ[2])
  uθd_rb = bstuθd'*ud

  function get_uθ_rb(x_rb::AbstractArray)
    ufe = reshape(bstu*x_rb[1:nstu],:,length(timesθ))
    uθfe = compute_in_timesθ(ufe,θ)
    bstuθ'*uθfe[:]
  end

  err = 1.
  iter = 0
  while norm(err) > tol && iter < maxit
    uθ_rb = get_uθ_rb(x_rb)
    jx_rb,rx_rb = jac(uθ_rb),res(uθ_rb,uθd_rb,x_rb)
    err = jx_rb \ rx_rb
    x_rb -= err
    iter += 1
    println("err = $(norm(err)), iter = $iter")
  end

  x_rb
end

function initial_guess(
  rbspace::NTuple{2,<:RBSpaceSteady},
  uh::Snapshots,
  ph::Snapshots,
  μ::Vector{Param},
  μk::Param)

  bsu,bsp = get_basis_space(rbspace)
  kmin = nearest_solution(μ,μk)
  x0 = vcat(bsu'*get_snap(uh[kmin])[:],bsp'*get_snap(ph[kmin])[:])
  x0
end

function initial_guess(
  rbspace::NTuple{2,<:RBSpaceUnsteady},
  uh::Snapshots,
  ph::Snapshots,
  μ::Vector{Param},
  μk::Param)

  bstu,bstp = get_basis_spacetime(rbspace)
  kmin = nearest_solution(μ,μk)
  x0 = vcat(bstu'*get_snap(uh[kmin])[:],bstp'*get_snap(ph[kmin])[:])
  x0
end

function nearest_solution(μ::Vector{Param},μk::Param)
  vars = [var(μi-μk) for μi=μ]
  argmin(vars)
end

function reconstruct_fe_sol(rbspace::RBSpaceSteady,rb_sol::Matrix{Float})
  bs = get_basis_space(rbspace)
  bs*rb_sol
end

function reconstruct_fe_sol(rbspace::RBSpaceUnsteady,rb_sol::Matrix{Float})
  bs = get_basis_space(rbspace)
  bt = get_basis_time(rbspace)
  ns = get_ns(rbspace)
  nt = get_nt(rbspace)

  rb_sol_resh = reshape(rb_sol,nt,ns)
  bs*(bt*rb_sol_resh)'
end

function reconstruct_fe_sol(rbspace::NTuple{2,RBSpaceSteady},rb_sol::Matrix{Float})
  bs_u,bs_p = get_basis_space.(rbspace)

  ns = get_ns.(rbspace)
  rb_sol_u,rb_sol_p = rb_sol[1:ns[1],:],rb_sol[1+ns[1]:end,:]

  bs_u*rb_sol_u,bs_p*rb_sol_p
end

function reconstruct_fe_sol(rbspace::NTuple{2,RBSpaceUnsteady},rb_sol::Matrix{Float})
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
