μ = realization(feop,10)

sols = collect_solutions(fesolver,feop,μ)
  # uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
  # ode_op = get_algebraic_operator(feop)
  # uu0 = get_free_dof_values(uh0(μ))
  # uμst = PODESolution(fesolver,ode_op,μ,uu0,t0,tf)
  # num_iter = Int(tf/fesolver.dt)
  # sols = allocate_solution(ode_op,num_iter)
  # for (u,t,n) in uμst
  #   printstyled("Computing fe solution at time $t for every parameter\n";color=:blue)
  #   sols[n] = get_solution(ode_op,u)
  # end
  # sols = Snapshots(sols)

function _get_gridap_sol(n::Int)
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

  results_ok = Vector{Float}[]
  for (uh,t) in sol_gridap
    push!(results_ok,copy(uh))
  end
  results_ok
end
N = 10
results_ok = _get_gridap_sol(N)
for i in eachindex(results_ok)
  println(isapprox(sols.snaps[i][N],results_ok[i]))
end

_u = sols[N]
for i in eachindex(results_ok)
  println(isapprox(_u[i],results_ok[i]))
end

# RES/JAC
μ = realization(feop,10)
sols = collect_solutions(fesolver,feop,μ)
times = get_times(fesolver)
_trian = Ω

nsnaps = info.nsnaps_system
snapsθ = recenter(fesolver,sols,μ)
[test_ptarray(snapsθ.snaps[i],sols.snaps[i]) for i = eachindex(snapsθ.snaps)]
_μ,_snapsθ = μ[1],snapsθ[1]
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,_μ,times)

dt,θ = fesolver.dt,fesolver.θ
dtθ = θ == 0.0 ? dt : dt*θ
ode_cache = update_cache!(ode_cache,ode_op,_μ,times)
nlop = PThetaMethodNonlinearOperator(ode_op,_μ,times,dtθ,_snapsθ,ode_cache,_snapsθ)
Xh, = ode_cache
dxh = ()
_xh = (_snapsθ,_snapsθ-_snapsθ)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh=TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))

b = allocate_residual(ode_op,_μ,times,_snapsθ,ode_cache)
vecdata = collect_cell_vector(test,feop.res(_μ,times,xh,dv),_trian)
assemble_vector_add!(b,feop.assem,vecdata)
A = allocate_jacobian(ode_op,_μ,times,_snapsθ,ode_cache)
matdata = collect_cell_matrix(trial(_μ,times),test,feop.jacs[1](_μ,times,xh,du,dv),_trian)
assemble_matrix_add!(A,feop.assem,matdata)

g_ok(x,t) = g(x,_μ,t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(test,g_ok)
m_ok(t,dut,v) = ∫(v*dut)dΩ
lhs_ok(t,du,v) = ∫(a(_μ,t)*∇(v)⋅∇(du))dΩ
rhs_ok(t,v) = ∫(f(_μ,t)*v)dΩ + ∫(h(_μ,t)*v)dΓn
feop_ok = TransientAffineFEOperator(m_ok,lhs_ok,rhs_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)

for (n,tn) in enumerate(times)
  xhF_ok = copy(_snapsθ[n]),0. * _snapsθ[n]
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,tn)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = ()
  for i in 2:2
    dxh_ok = (dxh_ok...,EvaluationFunction(Xh_ok[i],xhF_ok[i]))
  end
  xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],xhF_ok[1]),dxh_ok)
  vecdata_ok = collect_cell_vector(test,feop_ok.res(tn,xh_ok,dv),_trian)
  res_ok = assemble_vector(feop_ok.assem_t,vecdata_ok)
  matdata_ok = collect_cell_matrix(trial_ok(tn),test,feop_ok.jacs[1](tn,xh_ok,du,dv),_trian)
  jac_ok = assemble_matrix(feop_ok.assem_t,matdata_ok)
  test_ptarray(b,res_ok;n)
  test_ptarray(A,jac_ok;n)
end
