begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

begin
  mesh = "model_circle_2D_coarse.json"
  test_path = "$root/tests/navier-stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
  # mesh = "cube2x2.json"
  # bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
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
  # test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
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
  load_structures = false
  save_structures = true
  energy_norm = [:l2,:l2]
  compute_supremizers = true
  nsnaps_state = 50
  nsnaps_system = 30
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

rbspace = reduced_basis(info,feop,sols,params)
rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
save(info,(rbspace,rbrhs,rblhs))

snaps_test,params_test = load_test(info,feop,fesolver)
println("Solving nonlinear RB problems with Newton iterations")
xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1:1]
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,xn,μn)
nl_cache = nothing
x = map(zero,xn)
for iter in 1:fesolver.nls.max_nliters
  # x = recenter(fesolver,x,params_test)
  rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,x,μn)
  lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,x,μn)
  dxrb = NonaffinePTArray([lhs[1] \ rhs[1]])
  x -= recast(dxrb,rbspace)
  err = map(norm,dxrb.array)
  err_inf = norm(err)
  # println(norm(lhs[1]))
  # println(norm(rhs[1]))
  println("Iter $iter, error norm: $err_inf")
  # if err_inf ≤ fesolver.nls.tol
  #   return x
  # end
  # if iter == fesolver.nls.max_nliters
  #   @unreachable
  # end
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
R1((u,p),t) = (assemble_vector(dv -> ∫(dv⋅∂ₚt(u))dΩ,test_u)
  + assemble_vector(dv -> ∫(a(μn,t)*∇(dv)⊙∇(u))dΩ,test_u)
  + assemble_vector(dv -> c_ok(t,u,dv),test_u)
  - assemble_vector(dv -> ∫(p*(∇⋅(dv)))dΩ,test_u))
R2((u,p),t) = - assemble_vector(dq -> ∫(dq*(∇⋅(u)))dΩ,test_p)

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

function my_get_x(x,t)
  v0 = zero(x)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Xh_ok,_,_ = ode_cache_ok
  dxh_ok = (EvaluationFunction(Xh_ok[2],v0),)
  TransientCellField(EvaluationFunction(Xh_ok[1],x),dxh_ok)
end

function compute_ok_quantities(x)
  LHS111 = NnzMatrix([NnzVector(AA(t) + dC(my_get_x(u,t)[1],t)) for (u,t) in zip(x.array,times)]...)
  LHS112 = NnzMatrix([NnzVector(M) for k = 1:ntimes]...)
  LHS21 = NnzMatrix([NnzVector(B) for _ = 1:ntimes]...)
  LHS12 = NnzMatrix([NnzVector(sparse(B')) for _ = 1:ntimes]...)

  LHS111_rb = space_time_projection(LHS111,rbspace[1],rbspace[1])
  LHS112_rb = space_time_projection(LHS112,rbspace[1],rbspace[1];combine_projections=(x,y)->x-y)
  LHS11_rb = LHS111_rb + LHS112_rb
  LHS21_rb = space_time_projection(LHS21,rbspace[2],rbspace[1])
  LHS12_rb = space_time_projection(LHS12,rbspace[1],rbspace[2])

  R11 = NnzMatrix([R1(my_get_x(u,t),t) for (u,t) in zip(x.array,times)]...)
  R21 = NnzMatrix([R2(my_get_x(u,t),t) for (u,t) in zip(x.array,times)]...)
  RHS11_rb = space_time_projection(R11,rbspace[1])
  RHS21_rb = space_time_projection(R21,rbspace[2])

  return LHS11_rb,LHS21_rb,LHS12_rb,RHS11_rb,RHS21_rb
end

# xn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])]
xn = zero.([PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])])
x = vcat(xn...)
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,xn,Table([μn]))
rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,rbspace,xn,Table([μn]))
lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,xn,Table([μn]))

LHS11_rb,LHS21_rb,LHS12_rb,RHS11_rb,RHS21_rb = compute_ok_quantities(x)

nu = get_rb_ndofs(rbspace[1])
ℓ∞(lhs[1][1:nu,1:nu] - LHS11_rb)
ℓ∞(lhs[1][1+nu:end,1:nu] - LHS21_rb)
ℓ∞(lhs[1][1:nu,1+nu:end] - LHS12_rb)

ℓ∞(rhs[1][1:nu] - RHS11_rb)
ℓ∞(rhs[1][1+nu:end] - RHS21_rb)
