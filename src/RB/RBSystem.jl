function online_assembler(
  op::RBVarOperator,
  basis::NTuple{2,Matrix{Float}},
  μ::Param,
  args...)

  Broadcasting(b->online_assembler(op,b,μ))(basis)
end

function online_assembler(
  op::RBVarOperator,
  basis::Matrix{Float},
  μ::Param,
  args...)

  coeff = compute_coefficient(op,μ)
  online_structure(op,basis,coeff)
end

function online_assembler(
  op::RBVarOperator,
  mdeim::Union{MDEIM,NTuple{2,MDEIM}},
  μ::Param,
  args...)

  basis = get_basis_space(mdeim)
  coeff = compute_coefficient(op,mdeim,μ,args...)
  online_structure(op,basis,coeff)
end

function online_assembler(
  op::RBBilinOperator,
  basis_mdeim::Tuple{Matrix{Float},MDEIM},
  μ::Param,
  args...)

  affine_basis,mdeim = basis_mdeim
  affine_structure = online_assembler(op,affine_basis,μ)

  op_lift = RBLiftingOperator(op)
  nonaffine_basis = get_basis_space(mdeim)
  coeff = compute_coefficient(op_lift,mdeim,μ,args...)
  nonaffine_structure = online_structure(op_lift,nonaffine_basis,coeff)

  affine_structure,nonaffine_structure
end

function online_structure(
  op::RBSteadyVarOperator,
  basis::Union{Matrix{Float},NTuple{2,Matrix{Float}}},
  coeff)

  nr = get_nrows(op)
  basis_by_coeff_mult(basis,coeff,nr)
end

function online_structure(
  op::RBSteadyVarOperator{Nonlinear,<:ParamTrialFESpace},
  basis::Matrix{Float},
  coeff)

  nr = get_nrows(op)
  u -> basis_by_coeff_mult(basis,coeff(u),nr)
end

function online_structure(
  op::RBSteadyBilinOperator{Nonlinear,<:ParamTrialFESpace},
  basis::NTuple{2,Matrix{Float}},
  coeff)

  nr = get_nrows(op)
  M(u) = basis_by_coeff_mult(basis[1],coeff[1](u),nr)
  lift(u) = basis_by_coeff_mult(basis[2],coeff[2](u),nr)
  M,lift
end

function online_structure(
  op::RBUnsteadyVarOperator,
  basis::Union{Matrix{Float},NTuple{2,Matrix{Float}}},
  coeff)

  dtθ = get_dt(op)*get_θ(op)
  if get_id(op) == :M coeff = coeff ./ dtθ end

  btbtc = coeff_by_time_bases(op,coeff)
  ns_row = get_ns(get_rbspace_row(op))
  basis_block = blocks(basis,ns_row)

  nr = get_nrows(op)
  basis_by_coeff_mult(basis_block,btbtc,nr)
end

function online_structure(
  op::RBUnsteadyVarOperator{Nonlinear,<:ParamTransientTrialFESpace},
  basis::Matrix{Float},
  coeff)

  dtθ = get_dt(op)*get_θ(op)
  if get_id(op) == :M coeff = coeff ./ dtθ end

  btbtc = coeff_by_time_bases(op,coeff)
  ns_row = get_ns(get_rbspace_row(op))
  basis_block = blocks(basis,ns_row)

  nr = get_nrows(op)
  M(u) = basis_by_coeff_mult(basis_block,btbtc[1](u),nr)
  Mshift(u) = basis_by_coeff_mult(basis_block,btbtc[2](u),nr)
  M,Mshift
end

function online_structure(
  op::RBUnsteadyBilinOperator{Nonlinear,<:ParamTransientTrialFESpace},
  basis::NTuple{2,Matrix{Float}},
  coeff)

  dtθ = get_dt(op)*get_θ(op)
  if get_id(op) == :M coeff = coeff ./ dtθ end

  btbtc,btbtc_lift = coeff_by_time_bases(op,coeff)
  ns_row = get_ns(get_rbspace_row(op))
  basis_block = blocks(basis,ns_row)

  nr = get_nrows(op)
  M(u) = basis_by_coeff_mult(basis_block[1],btbtc[1](u),nr)
  Mshift(u) = basis_by_coeff_mult(basis_block[1],btbtc[2](u),nr)
  lift(u) = basis_by_coeff_mult(basis_block[2],btbtc_lift(u),nr)
  (M,Mshift),lift
end

function coeff_by_time_bases(
  op::Union{RBUnsteadyLinOperator,RBUnsteadyLiftingOperator},
  coeff)

  coeff_by_time_bases_lin(op,coeff)
end

function coeff_by_time_bases(
  op::RBUnsteadyBilinOperator,
  coeff)

  coeff_by_time_bases_bilin(op,coeff)
end

function coeff_by_time_bases(
  op::RBUnsteadyBilinOperator,
  coeff::NTuple{2,Any})

  @assert length(coeff) == 2 "Something is wrong"
  coeff_by_time_bases_bilin(op,first(coeff)),coeff_by_time_bases_lin(op,last(coeff))
end

function coeff_by_time_bases_lin(
  op::RBVarOperator,
  coeff::AbstractMatrix)

  rbrow = get_rbspace_row(op)
  rb_time_projection(rbrow,coeff)
end

function coeff_by_time_bases_lin(
  op::RBVarOperator{Nonlinear,<:ParamTransientTrialFESpace},
  coeff::Function)

  rbrow = get_rbspace_row(op)
  u->rb_time_projection(rbrow,coeff(u))
end

function coeff_by_time_bases_bilin(
  op::RBVarOperator,
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
  op::RBVarOperator{Nonlinear,<:ParamTransientTrialFESpace},
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
end

function poisson_rb_system(
  lhs::Matrix{Float},
  rhs::NTuple{N,Matrix{Float}}) where N

  A_rb = lhs
  F_rb,H_rb,lifts... = rhs
  A_rb,F_rb+H_rb-sum(lifts)
end

function poisson_rb_system(
  lhs::NTuple{4,Matrix{Float}},
  rhs::NTuple{N,Matrix{Float}},
  θ::Real) where N

  A_rb,Ashift_rb,M_rb,Mshift_rb = lhs
  F_rb,H_rb,lifts... = rhs

  rb_lhs = θ*(A_rb+M_rb) + (1-θ)*Ashift_rb - θ*Mshift_rb
  rb_rhs = F_rb+H_rb-sum(lifts)
  rb_lhs,rb_rhs
end

function stokes_rb_system(
  lhs::NTuple{2,Matrix{Float}},
  rhs::NTuple{N,Matrix{Float}}) where N

  A_rb,B_rb = lhs
  F_rb,H_rb,lifts... = rhs

  np = size(B_rb)[1]
  rb_lhs = vcat(hcat(A_rb,-B_rb'),hcat(B_rb,zeros(np,np)))
  rb_rhs = vcat(F_rb+H_rb-sum(lifts[1:end-1]),-lifts[end])
  rb_lhs,rb_rhs
end

function stokes_rb_system(
  lhs::NTuple{8,Matrix{Float}},
  rhs::NTuple{N,Matrix{Float}},
  θ::Real) where N

  A_rb,Ashift_rb,B_rb,Bshift_rb,BT_rb,BTshift_rb,M_rb,Mshift_rb = lhs
  F_rb,H_rb,lifts... = rhs

  np = size(B_rb,1)

  rb_lhs_11 = θ*(A_rb+M_rb) + (1-θ)*Ashift_rb - θ*Mshift_rb
  rb_lhs_12 = - θ*BT_rb - (1-θ)*BTshift_rb
  rb_lhs_21 = θ*B_rb + (1-θ)*Bshift_rb
  rb_lhs = vcat(hcat(rb_lhs_11,rb_lhs_12),hcat(rb_lhs_21,zeros(np,np)))

  rb_rhs = vcat(F_rb+H_rb-sum(lifts[1:end-1]),-lifts[end])
  rb_lhs,rb_rhs
end

function navier_stokes_rb_system(lhs::Tuple,rhs::Tuple)
  A_rb,B_rb,C_rb,D_rb = lhs
  F_rb,H_rb,lifts... = rhs
  liftA,liftB,liftC, = lifts

  lin_rb_lhs,lin_rb_rhs = stokes_rb_system((A_rb,B_rb),(F_rb,H_rb,liftA,liftB))

  nu,np = size(A_rb,1),size(B_rb,1)
  block12 = zeros(nu,np)
  block21 = zeros(np,nu)
  block22 = zeros(np,np)
  nonlin_rb_lhs1(u) = vcat(hcat(C_rb(u),block12),hcat(block21,block22))
  nonlin_rb_lhs2(u) = vcat(hcat(C_rb(u)+D_rb(u),block12),hcat(block21,block22))
  nonlin_rb_rhs(u) = vcat(liftC(u),zeros(np,1))

  jac_rb(u) = lin_rb_lhs + nonlin_rb_lhs2(u)
  lhs_rb(u) = lin_rb_lhs + nonlin_rb_lhs1(u)
  rhs_rb(u) = lin_rb_rhs + nonlin_rb_rhs(u)
  res_rb(u,ud,x_rb) = lhs_rb(u)*x_rb - rhs_rb(ud)

  res_rb,jac_rb
end

function navier_stokes_rb_system(lhs::Tuple,rhs::Tuple,θ::Real)
  A_rb,B_rb,BT_rb,C_rb,Cshift_rb,D_rb,Dshift_rb,M_rb = lhs
  F_rb,H_rb,lifts... = rhs
  liftA,liftB,liftC,liftM, = lifts

  lin_rb_lhs,lin_rb_rhs = stokes_rb_system((A_rb...,B_rb...,BT_rb...,M_rb...),
    (F_rb,H_rb,liftA,liftM,liftB),θ)

  nu,np = size(first(A_rb),1),size(first(B_rb),1)
  block12 = zeros(nu,np)
  block21 = zeros(np,nu)
  block22 = zeros(np,np)
  nonlin_rb_lhs1(u) = vcat(hcat(θ*C_rb(u) + (1-θ)*Cshift_rb(u),block12),
    hcat(block21,block22))
  nonlin_rb_lhs2(u) = vcat(hcat(θ*(C_rb(u)+D_rb(u)) +
    (1-θ)*(Cshift_rb(u)+Dshift_rb(u)),block12),hcat(block21,block22))
  nonlin_rb_rhs(u) = vcat(liftC(u),zeros(np,1))

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
  fespaces::NTuple{2,FESpace},
  rbspace::NTuple{2,RBSpace};
  tol=1e-10,maxit=10)

  println("Solving system via Newton method")

  Uk,Vk = fespaces
  bsu,bsp = get_basis_space(rbspace)
  nsu,nsp = size(bsu,2),size(bsp,2)
  x_rb = zeros(nsu+nsp,1)

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
  fespaces::Tuple{Function,FESpace},
  rbspace::NTuple{2,<:RBSpace},
  timesθ::Vector{<:Real},
  θ::Real,
  tol=1e-10,maxit=10)

  println("Solving system via Newton method")

  Uk,Vk = fespaces
  bstu = get_basis_spacetime(first(rbspace))
  bstp = get_basis_spacetime(last(rbspace))
  nstu,nstp = size(bstu,2),size(bstp,2)
  x_rb = zeros(nstu+nstp,1)

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
