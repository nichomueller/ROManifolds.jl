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

rbspace = reduced_basis(info,feop,sols,fesolver,μ)
  # nzm = NnzArray(sols)
  # basis_space = tpod(nzm,nothing;ϵ=1e-4)
  # compressed_nza = prod(basis_space,nzm)
  # compressed_nza_t = change_mode(compressed_nza)
  # basis_time = tpod(compressed_nza_t;ϵ=1e-4)
  # rbspace = RBSpace(basis_space,basis_time)

nzm = NnzArray(sols)
full_val = recast(nzm)
bs = rbspace.basis_space
bt = rbspace.basis_time
maximum(abs.(full_val - bs*bs'*full_val)) <= ϵ*10
m2 = change_mode(nzm)
maximum(abs.(m2 - bt*bt'*m2)) <= ϵ*10
