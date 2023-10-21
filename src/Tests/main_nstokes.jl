begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

begin
  mesh = "cube2x2.json"
  test_path = "$root/tests/navier-stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  order = 2
  degree = 4

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  m(μ,t,u,v) = ∫ₚ(v⋅u,dΩ)
  a(μ,t,(u,p),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
  c(μ,t,u,v) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
  dc(μ,t,u,du,v) = ∫ₚ(v⊙(dconv∘(du,∇(du),u,∇(u))),dΩ)

  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = m(μ,t,dut,v)
  jac(μ,t,(u,p),(du,dp),(v,q)) = a(μ,t,(du,dp),(v,q)) + dc(μ,t,u,du,v)
  res(μ,t,(u,p),(v,q)) = m(μ,t,∂ₚt(u),v) + a(μ,t,(u,p),(v,q)) + c(μ,t,u,v)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial_u = PTTrialFESpace(test_u,g)
  test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = PTFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.05,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
  ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
  xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))

  nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
  fesolver = PThetaMethod(nls,xh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = true
  save_solutions = true
  load_structures = true
  save_structures = true
  energy_norm = [:l2,:l2]
  compute_supremizers = true
  nsnaps_state = 50
  nsnaps_system = 20
  nsnaps_test = 10
  st_mdeim = false
  info = RBInfo(test_path;ϵ,load_solutions,save_solutions,load_structures,save_structures,
                energy_norm,compute_supremizers,nsnaps_state,nsnaps_system,nsnaps_test,st_mdeim)
  # multi_field_rb_model(info,feop,fesolver)
end

sols,params = load(info,(BlockSnapshots,Table))
rbspace = load(info,BlockRBSpace)
rbrhs,rblhs = load(info,(BlockRBVecAlgebraicContribution,Vector{BlockRBMatAlgebraicContribution}))

nsnaps = info.nsnaps_state
params = realization(feop,nsnaps)
trial = get_trial(feop)
sols,stats = collect_solutions(fesolver,feop,trial,params)
save(info,(sols,params,stats))

# rbspace = reduced_basis(info,feop,sols,params)
# rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
# save(info,(rbspace,rbrhs,rblhs))

# test_rb_solver(info,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)

snaps_test,params_test = load_test(info,feop,fesolver)
println("Solving nonlinear RB problems with Newton iterations")
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,snaps_test,params_test)
nl_cache = nothing
x = initial_guess(sols,params,params_test)
xrb = space_time_projection(x,rbspace)
for iter in 1:fesolver.nls.max_nliters
  x = recenter(fesolver,x,params_test)
  rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,x,params_test)
  lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,x,params_test)
  dxrb = PTArray([lhs[i] \ rhs[i] for i = eachindex(lhs)])
  xrb .+= dxrb
  x = recast(xrb,rbspace)
  err = map(norm,dxrb.array)
  err_inf = ℓ∞(err)
  println("Iter $iter, error norm: $err_inf")
  if err_inf ≤ fesolver.nls.tol
    return x
  end
  if iter == fesolver.nls.max_nliters
    @unreachable
  end
end
post_process(info,feop,fesolver,snaps_test,params_test,x,stats)

times = get_times(fesolver)
ntimes = length(times)

rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,snaps_test,params_test)
lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,snaps_test,params_test)

M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u) / (θ*dt)
AA(t) = assemble_matrix((du,dv)->∫(a(μn,t)*∇(dv)⊙∇(du))dΩ,trial_u(μn,t),test_u)
dC(u,t) = assemble_matrix((du,dv)->dc_ok(t,u,du,dv),trial_u(μn,t),test_u)
B = -assemble_matrix((du,dq)->∫(dq*(∇⋅(du)))dΩ,trial_u(μn,dt),test_p)

xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1]
g_ok(x,t) = g(x,μn,t)
g_ok(t) = x->g_ok(x,t)
m_ok(t,u,v) = ∫(v⋅u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μn,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
c_ok(t,u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
dc_ok(t,u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
Jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
Res_ok(t,(u,p),(v,q)) = m_ok(t,∂t(u),v) + a_ok(t,(u,p),(v,q)) + c_ok(t,u,v)
trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok = TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(Res_ok,Jac_ok,jac_t_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = allocate_cache(ode_op_ok)
xcat = vcat(xn...)
v0 = zero(xcat[1])
function _get_u(k)
  ukθ = xcat[k]
  t = times[k]
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
  TransientCellField(EvaluationFunction(Xh_ok[1],ukθ),dxh_ok)
end

LHS11 = NnzMatrix([NnzVector(M + AA(times[k]) + dC(_get_u(k)[1],times[k])) for k = 1:ntimes]...)
LHS21 = NnzMatrix([NnzVector(B) for _ = 1:ntimes]...)
LHS12 = NnzMatrix([NnzVector(sparse(B')) for _ = 1:ntimes]...)

LHS11_rb = space_time_projection(LHS11,rbspace[1],rbspace[1])
LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

nu = get_rb_ndofs(rbspace[1])
ℓ∞(lhs[1][1:nu,1:nu] - LHS11_rb)
ℓ∞(lhs[1][1+nu:end,1:nu] - LHS21_rb)
ℓ∞(lhs[1][1:nu,1+nu:end] - LHS12_rb)

# ARTIFICIAL PROBLEM
θdt = θ*dt
M = assemble_matrix((du,dv)->∫(dv⋅du)dΩ,trial_u(μn,dt),test_u)
Lu_l((u,p),t) = (assemble_vector(dv->∫(a(μn,t)*∇(dv)⊙∇(u))dΩ,test_u) + assemble_vector(dv->∫(dv⋅∂ₚt(u))dΩ,test_u) -
  assemble_vector(dv->∫(p*(∇⋅(dv)))dΩ,test_u))
Lu_nl(u,t) = assemble_vector(dv->c_ok(t,u,dv),test_u)
Lp(u,t) = -assemble_vector(dq->∫(dq*(∇⋅(u)))dΩ,test_p)
RHS((u,p),t) = vcat(Lu_nl(u,t) + Lu_l((u,p),t), Lp(u,t))
RHS_add(u) = vcat(M*u/θdt, zeros(Np))
LHS_l_11(t) = assemble_matrix((du,dv)->∫(a(μn,t)*∇(dv)⊙∇(du))dΩ,trial_u(μn,t),test_u) + M/θdt
LHS_l_12 = -assemble_matrix((dp,dv)->∫(dp*(∇⋅(dv)))dΩ,trial_p,test_u)
LHS_nl(u,t) = assemble_matrix((du,dv)->dc_ok(t,u,du,dv),trial_u(μn,t),test_u)
LHS(u,t) = vcat(hcat(LHS_l_11(t)+LHS_nl(u,t),LHS_l_12),hcat(LHS_l_12',sparse(zeros(Np,Np))))

x = vcat(xn...)
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,Table([μn]),times)
ode_cache = update_cache!(ode_cache,ode_op,Table([μn]),times)
ptb = allocate_residual(ode_op,Table([μn]),times,x,ode_cache)
ptA1 = allocate_jacobian(ode_op,Table([μn]),times,x,ode_cache)
ptA2 = allocate_jacobian(ode_op,Table([μn]),times,x,ode_cache)

Nu = length(get_free_dof_ids(test_u))
Np = length(get_free_dof_ids(test_p))

function prelim_test(x)
  v0 = copy(x) .* 0.
  update_cache!(ode_cache,ode_op,μn,times)

  residual!(ptb,ode_op,Table([μn]),times,(x,v0),ode_cache)
  jacobian!(ptA1,ode_op,Table([μn]),times,(x,v0),1,1.0,ode_cache)
  jacobian!(ptA2,ode_op,Table([μn]),times,(x,v0),2,1/(dt*θ),ode_cache)

  for (kt,t) in enumerate(times)
    Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
    Xh_ok,_,_ = ode_cache_ok
    dxh_ok = (EvaluationFunction(Xh_ok[2],v0[kt]),)
    xh_ok = TransientCellField(EvaluationFunction(Xh_ok[1],x[kt]),dxh_ok)
    @check ptA1[kt] + ptA2[kt] ≈ LHS(xh_ok[1],t)
    @check ptb[kt] ≈ RHS(xh_ok,t)
  end
end

prelim_test(vcat(xn...))

function test_get_quantities(x)
  v0 = copy(x) .* 0.
  update_cache!(ode_cache,ode_op,μn,times)

  residual!(ptb,ode_op,Table([μn]),times,(x,v0),ode_cache)
  b = hcat(ptb.array...)
  b1 = NnzMatrix(collect(eachcol(b[1:Nu,:]))...)
  b2 = NnzMatrix(collect(eachcol(b[1+Nu:end,:]))...)
  brb1 = space_time_projection(b1,rbspace[1])
  brb2 = space_time_projection(b2,rbspace[2])

  jacobian!(ptA1,ode_op,Table([μn]),times,(x,v0),1,1.0,ode_cache)
  jacobian!(ptA2,ode_op,Table([μn]),times,(x,v0),2,1/(dt*θ),ode_cache)
  A11 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu]),ptA1.array)...)
  A21 = NnzMatrix(map(x->NnzVector(x[1+Nu:end,1:Nu]),ptA1.array)...)
  A12 = NnzMatrix(map(x->NnzVector(x[1:Nu,1+Nu:end]),ptA1.array)...)
  M11 = NnzMatrix(map(x->NnzVector(x[1:Nu,1:Nu]),ptA2.array)...)

  Arb11 = space_time_projection(A11,rbspace[1],rbspace[1];combine_projections=(x,y)->θ*x+(1-θ)*y)
  Arb21 = space_time_projection(A21,rbspace[2],rbspace[1];combine_projections=(x,y)->θ*x+(1-θ)*y)
  Arb12 = space_time_projection(A12,rbspace[1],rbspace[2];combine_projections=(x,y)->θ*x+(1-θ)*y)
  Mrb11 = space_time_projection(M11,rbspace[1],rbspace[1];combine_projections=(x,y)->θ*x-θ*y)

  np = get_rb_ndofs(rbspace[2])
  Arb = vcat(hcat(Arb11+Mrb11,Arb12),hcat(Arb21,zeros(np,np)))
  brb = vcat(brb1,brb2)

  return Arb,brb
end

function test_newton(x0)
  nu = get_rb_ndofs(rbspace[1])
  xrb = space_time_projection(x0,rbspace)[1]
  x = vcat(x0...)
  for iter in 1:fesolver.nls.max_nliters
    lhs,rhs = test_get_quantities(x)
    dxrb = lhs \ rhs
    xrb .-= dxrb
    x = vcat(PTArray(recast(xrb[1:nu],rbspace[1])),PTArray(recast(xrb[1+nu:end],rbspace[2])))
    err = norm(dxrb)
    println("Iter $iter, error norm: $err")
    if err ≤ fesolver.nls.tol
      return x
    end
    if iter == fesolver.nls.max_nliters
      return x # @unreachable
    end
  end
end

x0 = [PTArray(zeros(xn[1])),PTArray(zeros(xn[2]))]
result = test_newton(x0)
DIOCAN
# # ANALIZING BLOCK 11: THIS WORKS
# xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1]

# Nu = length(get_free_dof_ids(test_u))
# Np = length(get_free_dof_ids(test_p))
# vθ = zeros(Nu+Np)
# nlop0 = Gridap.ODEs.ODETools.ThetaMethodNonlinearOperator(ode_op_ok,t0,dt*θ,vθ,ode_cache_ok,vθ)
# Aok = allocate_jacobian(nlop0,vθ)
# jac_ok = []
# for (kt,t) in enumerate(get_times(fesolver))
#   uk = vcat(xn...)[kt]
#   ukprev = kt > 1 ? vcat(xn...)[kt-1] : get_free_dof_values(xh0μ(μn))
#   ukθ = θ*uk + (1-θ)*ukprev
#   ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)

#   z = zero(eltype(Aok))
#   fillstored!(Aok,z)
#   jacobians!(Aok,ode_op_ok,t,(ukθ,vθ),(1.0,1/(dt*θ)),ode_cache_ok)
#   # jacobian!(Aok,ode_op_ok,t,(ukθ,vθ),1,(1.0,1/(dt*θ))[1],ode_cache_ok)

#   push!(jac_ok,copy(NnzVector(Aok[1:Nu,1:Nu])))
# end

# feop_row_col = feop[1,1]
# Aok = allocate_jacobian(nlop0,vθ)
# A0 = PTArray([Aok[1:Nu,1:Nu] for _ = 1:ntimes])
# jacs1,_ = collect_jacobians_for_trian(fesolver,feop_row_col,xn[1],Table([μn]),times;i=1)
# jacs2,_ = collect_jacobians_for_trian(fesolver,feop_row_col,xn[1],Table([μn]),times;i=2)
# idx1 = jacs1[1].nonzero_idx
# idx2 = jacs2[1].nonzero_idx
# _jac1 = collect_jacobians_for_idx!(A0,fesolver,feop_row_col,xn[1],Table([μn]),times,idx1;i=1)
# _jac2 = collect_jacobians_for_idx!(A0,fesolver,feop_row_col,xn[1],Table([μn]),times,idx2;i=2)
# jacs1plus2 = jacs1[1] + jacs2[1]
# _jacs1plus2 = _jac1 + _jac2

# nnzjac_ok = hcat(jac_ok...)
# ℓ∞(nnzjac_ok - jacs1plus2)
# ℓ∞(nnzjac_ok - _jacs1plus2)

# LHS11 ≈ nnzjac_ok
