# INDEX OF TEST
K = 2
μ = realization(feop,K)
times = get_times(fesolver)
Nt = length(times)
N = K*Nt
nfree = test.nfree

sols, = collect_solutions(fesolver,feop,μ)

_snaps = sols[1:K]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
ode_cache = update_cache!(ode_cache,ode_op,μ,times)
Xh, = ode_cache
dxh = ()
# _xh = (PTArray([ones(nfree) for _ = eachindex(_snaps)]),
#       PTArray([zeros(nfree) for _ = eachindex(_snaps)]))
_xh = copy(_snaps),zero(_snaps)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)

dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
b = allocate_residual(ode_op,μ,times,_snaps,ode_cache)
vecdata = collect_cell_vector(test,feop.res(μ,times,xh,dv))
assemble_vector_add!(b,feop.assem,vecdata)
A = allocate_jacobian(ode_op,μ,times,_snaps,1,ode_cache)
matdata = collect_cell_matrix(trial(μ,times),test,feop.jacs[1](μ,times,xh,du,dv))#,trian)
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

function gridap_res_jac_for_int(_sols,n::Int;zero_sol=false)
  p = μ[slow_idx(n,Nt)]
  t = times[fast_idx(n,Nt)]
  sols = _sols[fast_idx(n,Nt)]

  g_ok(x,t) = g(x,p,t)
  g_ok(t) = x->g_ok(x,t)
  a_ok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
  b_ok(t,v) = ∫(v*f(p,t))dΩ + ∫(v*h(p,t))dΓn
  m_ok(t,ut,v) = ∫(ut*v)dΩ

  trial_ok = TransientTrialFESpace(test,g_ok)
  feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  ode_cache_ok = allocate_cache(ode_op_ok)

  if zero_sol
    xhF_ok = copy(0. * sols),copy(0. * sols)
  else
    xhF_ok = copy(sols),copy(0. * sols)
  end
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
    @assert isapprox(sols.snaps[nt][np],sols_ok[nt]) "Failed with np = $np, nt = $nt"
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
snaps_test,params_test = sols[1:K],μ[1:K]
u,μ = PTArray(snaps_test[1:ntimes]),params_test[1]
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
ptA = allocate_jacobian(ode_op,params_test,times,snaps_test,1,ode_cache)
vθ = zero(snaps_test)
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
A = allocate_jacobian(nlop0,vθ)

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

nsnaps_test = 10
rbres,rbjac = rbrhs.affine_decompositions,rblhs[1].affine_decompositions
snaps_train,params_train = sols[1:nsnaps_test],params[1:nsnaps_test]
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)
op_offline = get_ptoperator(fesolver,feop,snaps_train,params_train)
rhs_cache,lhs_cache = allocate_cache(op,snaps_test)
rhs_mdeim_cache, = rhs_cache
rhs_collect_cache, = rhs_mdeim_cache
_res = collect_reduced_residuals!(rhs_collect_cache,op,rbres)

b,Mcache = rhs_collect_cache
dom = map(get_integration_domain,rbres)
meas = map(get_measure,dom)
times = map(get_times,dom)
common_time = union(times...)
x = _get_ptarray_at_time(op.u0,op.tθ,common_time)
_b = _get_ptarray_at_time(b,op.tθ,common_time)
# ress,trian = residual_for_trian!(_b,op,x,common_time,meas)
# residual_for_trian!(b,op.odeop,op.μ,op.tθ,(op.vθ,op.vθ),op.ode_cache,meas)
Xh, = op.ode_cache
dxh = (EvaluationFunction(Xh[2],op.vθ),)
xh = TransientCellField(EvaluationFunction(Xh[1],op.vθ),dxh)
V = get_test(feop)
v = get_fe_basis(V)
dc = res(op.μ,op.tθ,xh,v,meas)

_temp_res(μ,t,u,v) = ∫(v*∂ₚt(u))dΓn + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΓn - ∫(fμt(μ,t)*v)dΓn - ∫(hμt(μ,t)*v)dΩ
_temp_feop = AffinePTFEOperator(_temp_res,jac,jac_t,pspace,trial,test)
_temp_op = get_ptoperator(fesolver,_temp_feop,snaps_test,params_test)
_temp_res_full, = collect_residuals_for_trian(_temp_op)

_temp_feop = AffinePTFEOperator(Res,jac,jac_t,pspace,trial,test)
_temp_op = get_ptoperator(fesolver,_temp_feop,snaps_test,params_test)
_res = collect_reduced_residuals!(rhs_collect_cache,_temp_op,rbres)

# res_full, = collect_residuals_for_trian(op)
# res1(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ
# res2(μ,t,u,v) = -∫(hμt(μ,t)*v)dΓn
# times = op.tθ
# ode_op = get_algebraic_operator(feop)
# ode_cache = allocate_cache(ode_op,params_test,times)
# ode_cache = update_cache!(ode_cache,ode_op,params_test,times)
# ω = zero(snaps_test)
# Xh, = ode_cache
# dxh = ()
# dxh = (EvaluationFunction(Xh[2],ω),)
# xh = TransientCellField(EvaluationFunction(Xh[1],ω),dxh)
# b = allocate_residual(ode_op,params_test,times,ω,ode_cache)
# vecdata = collect_cell_vector(test,res1(params_test,times,xh,dv))
# assemble_vector_add!(b,feop.assem,vecdata)

# bmat = stack(b.array)
# bmat ≈ res_full[1]

form(μ,t,u,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
for n = 1:1000
  dc = form(params_test,times,xh,dv)
  trians = [get_domains(dc)...]
  @assert typeof(trians[2]) <: BoundaryTriangulation "Failed for n = $n"
end
