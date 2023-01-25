#= function online_assembler(
  rb_structure::Tuple,
  μ::Param,
  args...)

  ntup_rb_structure = expand(rb_structure)
  online_assembler(ntup_rb_structure,μ,args...)
end

function online_assembler(
  rb_structure::NTuple{N,RBOfflineStructure},
  μ::Param,
  args...) where N

  Broadcasting(rb->online_assembler(rb,μ,args...))(rb_structure)
end

function online_assembler(
  rb_structure::RBOfflineStructure,
  args...)

  op = get_op(rb_structure)
  os = get_offline_structure(rb_structure)
  online_assembler(op,os,args...)
end

function online_assembler(
  op::RBVariable,
  basis::Matrix{Float},
  args...)

  coeff = compute_coefficient(op)
  RBOnlineStructure(op,basis,coeff)
end

function online_assembler(
  op::RBVariable,
  mdeim::MDEIM,
  μ::Param,
  args...)

  basis = get_basis_space(mdeim)
  coeff = compute_coefficient(op,mdeim,μ,args...)
  RBOnlineStructure(op,basis,coeff)
end

function coeff_by_time_bases(op::RBUnsteadyVariable,coeff)
  coeff_by_time_bases_lin(op,coeff)
end

function coeff_by_time_bases(op::RBUnsteadyBilinVariable,coeff)
  coeff_by_time_bases_bilin(op,coeff)
end

function coeff_by_time_bases_lin(
  op::RBVariable,
  coeff::AbstractMatrix)

  rbrow = get_rbspace_row(op)
  rb_time_projection(rbrow,coeff)
end

function coeff_by_time_bases_lin(
  op::RBVariable{Nonlinear,<:ParamTransientTrialFESpace},
  coeff::Function)

  rbrow = get_rbspace_row(op)
  u->rb_time_projection(rbrow,coeff(u))
end

function coeff_by_time_bases_bilin(
  op::RBVariable,
  coeff::AbstractMatrix)

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  time_proj(idx1,idx2) = rb_time_projection(rbrow,rbcol,coeff,idx1,idx2)

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_backwards,idx_forwards = 1:Nt-1,2:Nt

  btbtc = time_proj(idx,idx)
  btbtc_shift = time_proj(idx_forwards,idx_backwards)
  btbtc,btbtc_shift
end

function coeff_by_time_bases_bilin(
  op::RBVariable{Nonlinear,<:ParamTransientTrialFESpace},
  coeff::Function)

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  time_proj(u,idx1,idx2) = rb_time_projection(rbrow,rbcol,coeff(u),idx1,idx2)

  Nt = get_Nt(op)
  idx = 1:Nt
  idx_backwards,idx_forwards = 1:Nt-1,2:Nt

  btbtc(u) = time_proj(u,idx,idx)
  btbtc_shift(u) = time_proj(u,idx_forwards,idx_backwards)
  btbtc,btbtc_shift
end =#

function steady_poisson_rb_system(rbos::NTuple{N,RBOnlineStructure}) where N
  lhs = eval_on_structure(rbos,:A)
  rhs = eval_on_structure(rbos,(:F,:H,:A_lift))
  lhs,sum(rhs)
end

function unsteady_poisson_rb_system(rbos::NTuple{N,RBOnlineStructure}) where N
  lhs = eval_on_structure(rbos,(:A,:M))
  rhs = eval_on_structure(rbos,(:F,:H,:A_lift,:M_lift))
  sum(lhs),sum(rhs)
end

function steady_stokes_rb_system(rbos::NTuple{N,RBOnlineStructure}) where N
  lhs = eval_on_structure(rbos,(:A,:B))
  rhs = eval_on_structure(rbos,(:F,:H,:A_lift,:B_lift))

  np = size(lhs[2],1)
  rb_lhs = vcat(hcat(lhs[1],-lhs[2]'),hcat(lhs[2],zeros(np,np)))
  rb_rhs = vcat(sum(rhs[1:end-1]),rhs[end])
  rb_lhs,rb_rhs
end

function unsteady_stokes_rb_system(rbos::NTuple{N,RBOnlineStructure}) where N
  lhs = eval_on_structure(rbos,(:A,:B,:BT,:M))
  rhs = eval_on_structure(rbos,(:F,:H,:A_lift,:B_lift,:M_lift))

  np = size(lhs[2],1)
  rb_lhs = vcat(hcat(lhs[1]+lhs[4],-lhs[3]),hcat(lhs[2],zeros(np,np)))
  rb_rhs = vcat(rhs[1]+rhs[2]+rhs[3]+rhs[5],rhs[4])
  rb_lhs,rb_rhs
end

function steady_navier_stokes_rb_system(rbos::NTuple{N,RBOnlineStructure}) where N
  lin_rb_lhs,lin_rb_rhs = steady_stokes_rb_system(rbos)
  nonlin_lhs = eval_on_structure(rbos,(:C,:D))
  nonlin_rhs = eval_on_structure(rbos,:C_lift)

  opA,opB = get_op(rbos,(:A,:B))
  rbu,rbp = get_rbspace_row(opA),get_rbspace_row(opB)
  nu,np = get_ns(rbu),get_ns(rbp)

  block12,block21,block22 = zeros(nu,np),zeros(np,nu),zeros(np,np)
  nonlin_rb_lhs1(u) = vcat(hcat(nonlin_lhs[1](u),block12),
                           hcat(block21,block22))
  nonlin_rb_lhs2(u) = vcat(hcat(nonlin_lhs[1](u)+nonlin_lhs[2](u),block12),
                           hcat(block21,block22))
  nonlin_rb_rhs(u) = vcat(nonlin_rhs(u),zeros(np,1))

  jac_rb(u) = lin_rb_lhs + nonlin_rb_lhs2(u)
  lhs_rb(u) = lin_rb_lhs + nonlin_rb_lhs1(u)
  rhs_rb(u) = lin_rb_rhs + nonlin_rb_rhs(u)
  res_rb(u,ud,x_rb) = lhs_rb(u)*x_rb - rhs_rb(ud)

  res_rb,jac_rb
end

function unsteady_navier_stokes_rb_system(rbos::NTuple{N,RBOnlineStructure}) where N
  lin_rb_lhs,lin_rb_rhs = unsteady_stokes_rb_system(rbos)
  nonlin_lhs = eval_on_structure(rbos,(:C,:D))
  nonlin_rhs = eval_on_structure(rbos,:C_lift)

  opA,opB = get_op(rbos,(:A,:B))
  rbu,rbp = get_rbspace_row(opA),get_rbspace_row(opB)
  nu,np = get_ns(rbu)*get_nt(rbu),get_ns(rbp)*get_nt(rbp)

  block12,block21,block22 = zeros(nu,np),zeros(np,nu),zeros(np,np)
  nonlin_rb_lhs1(u) = vcat(hcat(nonlin_lhs[1](u),block12),
                           hcat(block21,block22))
  nonlin_rb_lhs2(u) = vcat(hcat(nonlin_lhs[1](u)+nonlin_lhs[2](u),block12),
                           hcat(block21,block22))
  nonlin_rb_rhs(u) = vcat(nonlin_rhs(u),zeros(np,1))

  jac_rb(u) = lin_rb_lhs + nonlin_rb_lhs2(u)
  lhs_rb(u) = lin_rb_lhs + nonlin_rb_lhs1(u)
  rhs_rb(ud) = lin_rb_rhs + nonlin_rb_rhs(ud)
  res_rb(u,ud,x_rb) = lhs_rb(u)*x_rb - rhs_rb(ud)

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
  x0::Matrix{Float},
  fespaces::Tuple{Function,FESpace},
  rbspace::NTuple{2,<:RBSpace},
  timesθ::Vector{<:Real},
  θ::Real,
  tol=1e-10,maxit=10)

  println("Solving system via Newton method")

  Uk,Vk = fespaces
  bstu = get_basis_spacetime(rbspace[1])
  nstu = size(bstu,2)
  x_rb = x0

  function uθfe(x_rb::AbstractArray)
    ufe = reshape(bstu*x_rb[1:nstu],:,length(timesθ))
    compute_in_timesθ(ufe,θ)
  end

  function ufun(uθfe::AbstractArray)
    n(tθ) = findall(x -> x == tθ,timesθ)[1]
    tθ -> FEFunction(Vk,uθfe[:,n(tθ)])
  end

  function udfun(uθfe::AbstractArray)
    n(tθ) = findall(x -> x == tθ,timesθ)[1]
    tθ -> FEFunction(Uk(tθ),uθfe[:,n(tθ)])
  end

  err = 1.
  iter = 0
  while norm(err) > tol && iter < maxit
    uθh = uθfe(x_rb)
    jx_rb,rx_rb = jac(ufun(uθh)),res(ufun(uθh),udfun(uθh),x_rb)
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
  x0 = vcat(bsu'*uh[kmin],bsp'*ph[kmin])
  Matrix(x0)
end

function initial_guess(
  rbspace::NTuple{2,<:RBSpaceUnsteady},
  uh::Snapshots,
  ph::Snapshots,
  μ::Vector{Param},
  μk::Param)

  bstu,bstp = get_basis_spacetime(rbspace)
  kmin = nearest_solution(μ,μk)
  x0 = vcat(bstu'*uh[kmin][:],bstp'*ph[kmin][:])
  Matrix(x0)
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
