K = 2
μ = realization(feop,K)
dtθ = dt*θ

# POISSON
ode_op = get_algebraic_operator(feop)
w = get_free_dof_values(uh0μ(μ))
sol = PODESolution(fesolver,ode_op,μ,w,t0,tf)

results = PTArray[]
for (uh,t) in sol
  ye = copy(uh)
  push!(results,ye)
end

n = 2
p,v = μ[n],w[n]
g_ok(x,t) = g(x,p,t)
g_ok(t) = x->g_ok(x,t)
a_ok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
b_ok(t,v) = ∫(v*f(p,t))dΩ + ∫(v*h(p,t))dΓn
m_ok(t,ut,v) = ∫(ut*v)dΩ

trial_ok = TransientTrialFESpace(test,g_ok)
feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_solver = ThetaMethod(LUSolver(),dt,θ)
sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,v,t0,tf)

results_ok = Vector{Float}[]
for (uh,t) in sol_gridap
  ye = copy(uh)
  push!(results_ok,ye)
end

for i in eachindex(results)
  test_ptarray(results[i],results_ok[i];n)
end

boh = PTArray[]
@time for (uh,t) in sol
  push!(boh,uh)
end

boh_ok = Vector{Float}[]
@time for (uh,t) in sol_gridap
  push!(boh_ok,uh)
end

Aμ(t) = assemble_matrix((u,v) -> ∫(a(p,t)*∇(v)⋅∇(u))dΩ,trial(p,t),test)
Mμ(t) = assemble_matrix((u,v) -> ∫(v*u)dΩ,trial(p,t),test)
Fμ(t) = assemble_vector(v -> ∫(v*f(p,t))dΩ,test)
Hμ(t) = assemble_vector(v -> ∫(v*h(p,t))dΓn,test)

Lμ(t) = (assemble_vector(v->∫(a(p,t)*∇(v)⋅∇(zero(trial(p,t))))dΩ,test)
 + assemble_vector(v->∫(v*zero(∂ₚt(trial)(p,t)))dΩ,test))

for (nt,t) in enumerate(get_times(fesolver))
  un = results_ok[nt]
  unprev = nt > 1 ? results_ok[nt-1] : get_free_dof_values(uh0μ(p))
  Jn = Aμ(t) + Mμ(t)/dtθ
  rn = Fμ(t) + Hμ(t) - Lμ(t) + Mμ(t)*unprev/dtθ #- Lμ(t,un,unprev)
  @assert Jn \ rn ≈ un "Failed when n = $nt"
end

M = assemble_matrix((du,dv)->∫(dv*du)dΩ,trial(rand(3),dt),test)/(dt*θ)
vθ = zeros(test.nfree)
ode_cache = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dtθ,vθ,ode_cache,vθ)
b = allocate_residual(nlop0,vθ)
bok = copy(b)
A = allocate_jacobian(nlop0,vθ)
Aok = copy(A)
for (nt,t) in enumerate(get_times(fesolver))
  un = results_ok[nt]
  unprev = nt > 1 ? results_ok[nt-1] : get_free_dof_values(uh0μ(p))
  ode_cache = Gridap.ODEs.TransientFETools.update_cache!(ode_cache,ode_op_ok,t)
  nlop = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t,dtθ,unprev,ode_cache,vθ)
  z = zero(eltype(A))
  fillstored!(A,z)
  fill!(b,z)
  residual!(b,ode_op_ok,t,(vθ,vθ),ode_cache)
  jacobians!(A,ode_op_ok,t,(vθ,vθ),(1.0,1/dtθ),ode_cache)
  # Gridap.ODEs.ODETools._vector!(bok,ode_op_ok,t,dtθ,unprev,ode_cache,vθ)
  # Gridap.ODEs.ODETools._matrix!(Aok,ode_op_ok,t,dtθ,unprev,ode_cache,vθ)
  @assert A \ (M*unprev - b) ≈ un "Failed when n = $nt"
  # println("t = $t, diff res = $(ℓ∞(b + bok)), diff jac = $(ℓ∞(A - Aok))")
end
