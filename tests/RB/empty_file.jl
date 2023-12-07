times = get_times(fesolver)
ntimes = length(times)
xn,μn = PTArray(snaps_test[1:ntimes]),params_test[1]

g0_ok(x,t) = g0(x,μn,t)
g0_ok(t) = x->g0_ok(x,t)
g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
m_ok(t,u,v) = ∫(v⋅u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
c_ok(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ
dc_ok(t,u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ + ∫(v⊙(∇(u)'⋅du))dΩ

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
Jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
Res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,u,v)
trial_u_ok = TransientTrialFESpace(test_u,[g0_ok,g_ok])
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(Res_ok,Jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)

times = get_times(fesolver)
kt = 1
t = times[kt]
v0 = zero(xn[1])
x = kt > 1 ? xn[kt-1] : get_free_dof_values(xh0μ(μn))
Nu,Np = test_u.nfree,length(get_free_dof_ids(test_p))
Nt = num_time_dofs(fesolver)
θdt = θ*dt
vθ = zeros(Nu+Np)
ode_cache_ok = TransientFETools.allocate_cache(ode_op_ok)
nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,vθ,ode_cache_ok,vθ)
bok = allocate_residual(nlop0,vθ)
Aok = allocate_jacobian(nlop0,vθ)

M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u)

function return_quantities(ode_op,ode_cache,x::PTArray,kt)
  xk = x[kt]
  t = times[kt]
  ode_cache = TransientFETools.update_cache!(ode_cache,ode_op,t)

  z = zero(eltype(Aok))
  fillstored!(Aok,z)
  fill!(bok,z)
  residual!(bok,ode_op,t,(xk,v0),ode_cache)
  jacobians!(Aok,ode_op,t,(xk,v0),(1.0,1/(dt*θ)),ode_cache)
  return bok,Aok
end

x = copy(xn) .* 0.
nu = num_rb_ndofs(rbspace[1])
# WORKS!!!!!!!!!!
for iter in 1:fesolver.nls.max_nliters
  xrb = space_time_projection(map(x->x[1:Nu],x),rbspace[1]),space_time_projection(map(x->x[1+Nu:end],x),rbspace[2])
  A,b = [],[]
  for kt = eachindex(times)
    bk,Ak = return_quantities(ode_op_ok,ode_cache_ok,x,kt)
    push!(b,copy(bk))
    push!(A,copy(Ak))
  end

  LHS11_1 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu] - M/θdt),A)...)
  LHS11_2 = NnzMatrix([NnzVector(M/θdt) for _ = times]...)
  LHS21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),A)...)
  LHS12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),A)...)

  LHS11_rb_1 = space_time_projection(LHS11_1,rbspace[1],rbspace[1])
  LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
  LHS11_rb = LHS11_rb_1 + LHS11_rb_2
  LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
  LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

  np = num_rb_ndofs(rbspace[2])
  LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

  R1 = NnzMatrix(map(x->x[1:Nu],b)...)
  R2 = NnzMatrix(map(x->x[1+Nu:end],b)...)
  RHS1_rb = space_time_projection(R1,rbspace[1])
  RHS2_rb = space_time_projection(R2,rbspace[2])
  _RHS_rb = vcat(RHS1_rb,RHS2_rb)
  RHS_rb = vcat(LHS11_rb_2*vcat(xrb...)[1][1:nu],zeros(np)) + _RHS_rb

  println("Norm (RHS1,LHS11) = ($(RHS1_rb[1]),$(norm(LHS11_rb[1])))")
  dxrb = LHS_rb \ RHS_rb
  dxrb_1,dxrb_2 = dxrb[1:nu],dxrb[1+nu:end]
  xiter = vcat(recast(PTArray(dxrb_1),rbspace[1]),recast(PTArray(dxrb_2),rbspace[2]))
  x -= xiter

  nerr = norm(dxrb)
  println("norm dx = $nerr")
  if nerr ≤ 1e-10
    break
  end
end

norm(hcat(xn.array...) - hcat(x.array...)) / norm(hcat(xn.array...))

# COMPARISON
x = copy(xn) #.* 0.
xrb = space_time_projection(x,op,rbspace)
A,b = [],[]
for kt = eachindex(times)
  bk,Ak = return_quantities(ode_op_ok,ode_cache_ok,x,kt)
  push!(b,copy(bk))
  push!(A,copy(Ak))
end

LHS11_1 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu] - M/θdt),A)...)
LHS11_2 = NnzMatrix([NnzVector(M/θdt) for _ = times]...)
LHS21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),A)...)
LHS12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),A)...)

LHS11_rb_1 = space_time_projection(LHS11_1,rbspace[1],rbspace[1])
LHS11_rb_2 = space_time_projection(LHS11_2,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
LHS11_rb = LHS11_rb_1 + LHS11_rb_2
LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

np = num_rb_ndofs(rbspace[2])
LHS_rb = vcat(hcat(LHS11_rb,LHS12_rb),hcat(LHS21_rb,zeros(np,np)))

R1 = NnzMatrix(map(x->x[1:Nu],b)...)
R2 = NnzMatrix(map(x->x[1+Nu:end],b)...)
RHS1_rb = space_time_projection(R1,rbspace[1])
RHS2_rb = space_time_projection(R2,rbspace[2])
RHS_rb = vcat(RHS1_rb,RHS2_rb)
_RHS_rb = vcat(LHS11_rb_2*xrb[1][1:nu],zeros(np)) + RHS_rb

op = get_ptoperator(fesolver,feop,x,Table([μn]))
lrhs = collect_rhs_contributions!(rhs_cache,rbinfo,op,rbrhs,rbspace)
llhs = collect_lhs_contributions!(lhs_cache,rbinfo,op,rblhs,rbspace)
nlrhs = collect_rhs_contributions!(rhs_cache,rbinfo,op,nl_rbrhs,rbspace)
nllhs = collect_lhs_contributions!(lhs_cache,rbinfo,op,nl_rblhs,rbspace)
lhs = llhs + nllhs
rhs = llhs*xrb + (lrhs+nlrhs)
norm(LHS_rb - lhs[1],Inf)
norm(RHS_rb - rhs[1],Inf)
norm(_RHS_rb - rhs[1],Inf)
