# INDEX OF TEST
K = 2
μ = realization(op,K)
times = get_times(fesolver)
Nt = length(times)
N = K*Nt
nfree = test.nfree

sols = collect_solutions(fesolver,feop,μ)

snapsθ = recenter(fesolver,sols,μ)
[test_ptarray(snapsθ.snaps[i],sols.snaps[i]) for i = eachindex(snapsθ.snaps)]

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
  sols = ones(nfree)

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
    # test_ptarray(b,res_ok;n)
    # test_ptarray(A,jac_ok;n)
  end
end

np = 1
nt = 2
n = (np-1)*Nt+nt
xh_ok,res_ok,jac_ok = gridap_res_jac_for_int(sols_ok,n)

# # MODE2
# nzm = NnzArray(sols)
# m2 = change_mode(nzm)
# m2_ok = hcat(sols_ok...)'
# space_ndofs = size(m2_ok,2)
# @assert isapprox(m2_ok,m2[:,(N-1)*space_ndofs+1:N*space_ndofs])

# INVESTIGATE JACOBIAN
solsθ = recenter(fesolver,sols,μ)[1:N]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ[1:N],times)
ode_cache = update_cache!(ode_cache,ode_op,μ[1:N],times)
Xh, = ode_cache
dxh = ()
_xh = (solsθ,solsθ-solsθ)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)

# DOMAIN CONTRIB
dc = integrate(feop.jacs[1](μ[1:N],times,xh,du,dv))
dc_ok = [∫(a(p,t)*∇(dv)⋅∇(du))dΩ for p in μ for t in times]
for n in eachindex(dc_ok)
  test_ptarray(dc[Ω],dc_ok[n][Ω];n)
end

# CELL DATA
g_ok(x,t) = g(x,rand(3),t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(test,g_ok)
matdata = collect_cell_matrix(trial(μ,times),test,dc)
ntimes = length(times)
for n in eachindex(dc_ok)
  matdata_ok = collect_cell_matrix(trial_ok(times[fast_idx(n,ntimes)]),test,dc_ok[n])
  test_ptarray(matdata[1][1],matdata_ok[1][1];n)
end

# ALGEBRAIC STRUCTURE
A = allocate_jacobian(ode_op,μ,times,solsθ,ode_cache)
A0 = copy(A[1])
assemble_matrix_add!(A,feop.assem,matdata)
for n in eachindex(dc_ok)
  matdata_ok = collect_cell_matrix(trial_ok(times[fast_idx(n,ntimes)]),test,dc_ok[n])
  A_ok = copy(A0)
  assemble_matrix_add!(A_ok,feop.assem,matdata_ok)
  test_ptarray(A,A_ok;n)
end

# INVESTIGATE RESIDUAL
solsθ = recenter(fesolver,sols,μ)[N:N]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ[N:N],times)
ode_cache = update_cache!(ode_cache,ode_op,μ[N:N],times)
Xh, = ode_cache
dxh = ()
_xh = (solsθ,solsθ-solsθ)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)

# DOMAIN CONTRIB
res_fun(p,t,u,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) #- ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn) #∫ₚ(v*∂ₚt(u),dΩ) +
dc = integrate(res_fun(μ[N:N],times,xh,dv))
dc_ok = []
res_ok_fun(p,t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ #- ∫(v*f(p,t))dΩ - ∫(v*h(p,t))dΓn # ∫(∂t(u)*v)dΩ +
for (nt,t) in enumerate(times)
  ode_op_ok = get_algebraic_operator(feop_ok)
  ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok, = ode_cache_ok
  dxh_ok = ()
  _xh_ok = (sols_ok[nt],sols_ok[nt]-sols_ok[nt])
  for i in 2:get_order(feop)+1
    dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],_xh_ok[i]))
  end
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],_xh_ok[1]),dxh_ok)
  push!(dc_ok,res_ok_fun(μ[N],t,xh_ok,dv))
end
for n in eachindex(dc_ok)
  test_ptarray(dc[Ω],dc_ok[n][Ω];n)
end

dc1 = collect(dc[Ω][1])
dc_ok1 = collect(dc_ok[1][Ω])


# CELL DATA
g_ok(x,t) = g(x,rand(3),t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(test,g_ok)
matdata = collect_cell_matrix(trial(μ,times),test,dc)
ntimes = length(times)
for n in eachindex(dc_ok)
  matdata_ok = collect_cell_matrix(trial_ok(times[fast_idx(n,ntimes)]),test,dc_ok[n])
  test_ptarray(matdata[1][1],matdata_ok[1][1];n)
end

# ALGEBRAIC STRUCTURE
A = allocate_jacobian(ode_op,μ,times,solsθ,ode_cache)
A0 = copy(A[1])
assemble_matrix_add!(A,feop.assem,matdata)
for n in eachindex(dc_ok)
  matdata_ok = collect_cell_matrix(trial_ok(times[fast_idx(n,ntimes)]),test,dc_ok[n])
  A_ok = copy(A0)
  assemble_matrix_add!(A_ok,feop.assem,matdata_ok)
  test_ptarray(A,A_ok;n)
end










# @check all([A[n] == assemble_matrix(∫(a(params[slow_idx(n,ntimes)],times[fast_idx(n,ntimes)])*∇(dv)⋅∇(du))dΩ,
#   trial_ok(times[fast_idx(n,ntimes)]),test) for n = 1:100])
nparams = length(μ)
@check all([A[n] == assemble_matrix(∫(a(μ[fast_idx(n,nparams)],times[slow_idx(n,nparams)])*∇(dv)⋅∇(du))dΩ,
  trial_ok(times[slow_idx(n,nparams)]),test) for n = 1:100])

n = 1
for p in μ, t in times
  Aok = assemble_matrix(∫(a(p,t)*∇(dv)⋅∇(du))dΩ,trial_ok(t),test)
  A_ok = copy(A0)
  matdata_ok = collect_cell_matrix(trial_ok(t),test,∫(a(p,t)*∇(dv)⋅∇(du))dΩ)
  assemble_matrix_add!(A_ok,feop.assem,matdata_ok)
  @check Aok == A_ok
  @check A[n] == Aok "not true for n = $n"
  n += 1
end

μ = μ[1:2]
solsθ = recenter(fesolver,sols,μ)[1:2]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
ode_cache = update_cache!(ode_cache,ode_op,μ,times)
Xh, = ode_cache
dxh = ()
_xh = (solsθ,solsθ-solsθ)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)
A = allocate_jacobian(ode_op,μ,times,solsθ,ode_cache)
assemble_matrix_add!(A,feop.assem,matdata)

for (n,t) in enumerate(times)
  @check assemble_matrix(∫(a(μ[1],t)*∇(dv)⋅∇(du))dΩ,trial_ok(t),test) == A[n]
end

for (n,t) in enumerate(times)
  @check assemble_matrix(∫(a(μ[2],t)*∇(dv)⋅∇(du))dΩ,trial_ok(t),test) == A[ntimes+n]
end
