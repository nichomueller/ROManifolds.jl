# INDEX OF TEST
K = 2
μ = realization(feop,K)
times = get_times(fesolver)
Nt = length(times)
N = K*Nt
nfree = test.nfree

sols, = collect_solutions(fesolver,feop,get_trial(feop),μ)

snapsθ = recenter(fesolver,sols,μ)
# [test_ptarray(snapsθ.snaps[i],sols.snaps[i]) for i = eachindex(snapsθ.snaps)]

_snapsθ = snapsθ[1:K]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
ode_cache = update_cache!(ode_cache,ode_op,μ,times)
Xh, = ode_cache
dxh = ()
# _xh = (PTArray([ones(nfree) for _ = eachindex(_snapsθ)]),
#       PTArray([zeros(nfree) for _ = eachindex(_snapsθ)]))
_xh = copy(_snapsθ),_snapsθ-_snapsθ
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)

dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
b = allocate_residual(ode_op,μ,times,_snapsθ,ode_cache)
vecdata = collect_cell_vector(test,integrate(feop.res(μ,times,xh,dv)))#,trian)
assemble_vector_add!(b,feop.assem,vecdata)
A = allocate_jacobian(ode_op,μ,times,_snapsθ,ode_cache)
matdata = collect_cell_matrix(trial(μ,times),test,integrate(feop.jacs[1](μ,times,xh,du,dv)))#,trian)
assemble_matrix_add!(A,feop.assem,matdata)

function gridap_solutions_for_int(n::Int)
  p = μ[n]

  g_ok(x,t) = g(x,p,t)
  g_ok(t) = x->g_ok(x,t)
  a_ok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
  b_ok(t,v) = ∫(v*f(p,t))dΩ + ∫(v*h(p,t))dΓn
  m_ok(t,ut,v) = ∫(ut*v)dΩ

  trial_ok = TransientTrialFESpace(test,g_ok)
  feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  w0 = get_free_dof_values(uh0μ(p))
  ode_solver = ThetaMethod(LUSolver(),dt,θ)
  sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,w0,t0,tf)

  sols_ok = []
  for (uh,t) in sol_gridap
    push!(sols_ok,copy(uh))
  end

  sols_ok
end

function gridap_res_jac_for_int(_sols,n::Int)
  p = μ[slow_idx(n,Nt)]
  t = times[fast_idx(n,Nt)]
  sols = _sols[fast_idx(n,Nt)]
  # sols = ones(nfree)

  g_ok(x,t) = g(x,p,t)
  g_ok(t) = x->g_ok(x,t)
  a_ok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
  b_ok(t,v) = ∫(v*f(p,t))dΩ + ∫(v*h(p,t))dΓn
  m_ok(t,ut,v) = ∫(ut*v)dΩ

  trial_ok = TransientTrialFESpace(test,g_ok)
  feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  ode_cache_ok = allocate_cache(ode_op_ok)

  xhF_ok = copy(sols),copy(0. * sols)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = ()
  for i in 2:2
    dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],xhF_ok[i]))
  end
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],xhF_ok[1]),dxh_ok)
  vecdata_ok = collect_cell_vector(test,feop_ok.res(t,xh_ok,dv))#,trian)
  res_ok = assemble_vector(feop_ok.assem_t,vecdata_ok)
  matdata_ok = collect_cell_matrix(trial_ok(t),test,feop_ok.jacs[1](t,xh_ok,du,dv))#,trian)
  jac_ok = assemble_matrix(feop_ok.assem_t,matdata_ok)

  xh_ok,res_ok,jac_ok
end

for np = 1:K
  sols_ok = gridap_solutions_for_int(np)
  for nt in eachindex(sols_ok)
    @assert isapprox(sols.snaps[nt][np],sols_ok[nt]) "Failed with np = $n, nt = $nt"
  end
end

for np = 1:K
  sols_ok = gridap_solutions_for_int(np)
  for nt = 1:Nt
    n = (np-1)*Nt+nt
    xh_ok,res_ok,jac_ok = gridap_res_jac_for_int(sols_ok,n)
    test_ptarray(xh.cellfield.dirichlet_values,xh_ok.cellfield.dirichlet_values;n)
    test_ptarray(xh.cellfield.cell_dof_values,xh_ok.cellfield.cell_dof_values;n)
    test_ptarray(b,res_ok;n)
    test_ptarray(A,jac_ok;n)
  end
end

# MODE2
# nzm = NnzArray(sols)
# m2 = change_mode(nzm)
# m2_ok = hcat(sols_ok...)'
# space_ndofs = size(m2_ok,2)
# @assert isapprox(m2_ok,m2[:,(K-1)*space_ndofs+1:K*space_ndofs])

times = get_times(fesolver)
ntimes = length(times)
u,μ = snaps_test[1:ntimes],params_test[1]
g_ok(x,t) = g(x,μ,t)
g_ok(t) = x->g_ok(x,t)
a_ok(t,u,v) = ∫(a(μ,t)*∇(v)⋅∇(u))dΩ
b_ok(t,v) = ∫(v*f(μ,t))dΩ + ∫(v*h(μ,t))dΓn
m_ok(t,ut,v) = ∫(ut*v)dΩ

trial_ok = TransientTrialFESpace(test,g_ok)
feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,params_test,times)
ode_cache = update_cache!(ode_cache,ode_op,params_test,times)
ptb = allocate_residual(ode_op,params_test,times,snaps_test,ode_cache)
ptA = allocate_jacobian(ode_op,params_test,times,snaps_test,ode_cache)
vθ = copy(snaps_test) .* 0.
nlop = get_ptoperator(ode_op,params_test,times,dt*θ,snaps_test,ode_cache,vθ)
residual!(ptb,nlop,copy(snaps_test))
jacobian!(ptA,nlop,copy(snaps_test))
ptb1 = ptb[1:ntimes]
ptA1 = ptA[1:ntimes]

dtθ = dt*θ
M = assemble_matrix((du,dv)->∫(dv*du)dΩ,trial(μ,dt),test)/dtθ
vθ = zeros(test.nfree)
ode_cache = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dtθ,vθ,ode_cache,vθ)
b = allocate_residual(nlop0,vθ)
bok = copy(b)
A = allocate_jacobian(nlop0,vθ)
Aok = copy(A)

for (nt,t) in enumerate(get_times(fesolver))
  un = u[nt]
  unprev = nt > 1 ? u[nt-1] : get_free_dof_values(uh0μ(μ))
  ode_cache = Gridap.ODEs.TransientFETools.update_cache!(ode_cache,ode_op_ok,t)
  nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t,dtθ,unprev,ode_cache,vθ)
  z = zero(eltype(A))
  fillstored!(A,z)
  fill!(b,z)
  residual!(b,ode_op_ok,t,(vθ,vθ),ode_cache)
  jacobians!(A,ode_op_ok,t,(vθ,vθ),(1.0,1/dtθ),ode_cache)
  @assert b ≈ ptb1[nt] "Failed when n = $nt"
  @assert A ≈ ptA1[nt] "Failed when n = $nt"
  @assert A \ (M*unprev - b) ≈ θ*un + (1-θ)*unprev "Failed when n = $nt"
end

# DIFFERENT TEST
g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
m_ok(t,u,v) = ∫(v*u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⋅∇(u))dΩ

jac_t_ok(t,u,dut,v) = m_ok(t,dut,v)
jac_ok(t,u,du,v) = a_ok(t,(du,dp),(v,q))
res_ok(t,u,v) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q))
trial_ok = TransientTrialFESpace(test,g_ok)
feop_ok = TransientFEOperator(res_ok,jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)

function my_get_u(u,t)
  v0 = zero(u)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
  TransientCellField(EvaluationFunction(Xh_ok[1],v0),dxh_ok)
end

sols,params = load(info,(Snapshots,Table))
rbspace = load(info,RBSpace)
rbrhs,rblhs = load(info,(RBVecAlgebraicContribution,Vector{RBMatAlgebraicContribution}))
snaps_test,params_test = load_test(info,feop,fesolver)
x = nearest_neighbor(sols,params,params_test)
x .= recenter(fesolver,x,params)
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,snaps_test,params_test)
rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,x,params_test)
lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,x,params_test)

times = get_times(fesolver)
ntimes = length(times)
xn = PTArray(snaps_test[1:ntimes])
μn = params_test[1]
M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial(μn,dt),test) / (θ*dt)
AA(t) = assemble_matrix((du,dv)->∫(a(μn,t)*∇(dv)⊙∇(du))dΩ,trial(μn,t),test)
R(u,t) = (assemble_vector(dv -> ∫(dv*∂ₚt(u))dΩ,test)
  + assemble_vector(dv -> ∫(a(μn,t)*∇(dv)⋅∇(u))dΩ,test)
  - assemble_vector(dv -> ∫(f(μn,t)*dv)dΩ,test)
  - assemble_vector(dv -> ∫(h(μn,t)*dv)dΓn,test))

LHS1 = NnzMatrix([NnzVector(AA(t)) for t in times]...)
LHS2 = NnzMatrix([NnzVector(M) for t in times]...)
RHS = NnzMatrix([R(my_get_u(u,t),t) for (u,t) in zip(xn.array,times)]...)
LHS1_rb = space_time_projection(LHS1,rbspace,rbspace;combine_projections=(x,y)->θ*x+θ*y)
LHS2_rb = space_time_projection(LHS2,rbspace,rbspace;combine_projections=(x,y)->θ*x-θ*y)
LHS_rb = LHS1_rb + LHS2_rb
RHS_rb = space_time_projection(RHS,rbspace)

norm(lhs[1] - LHS_rb)
norm(rhs[1] - RHS_rb)
