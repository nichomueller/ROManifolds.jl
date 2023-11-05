begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

begin
  mesh = "cube2x2.json"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  # mesh = "model_circle_2D_coarse.json"
  # bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
  test_path = "$root/tests/navier-stokes/unsteady/$mesh"
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

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
  dc(u,du,v) = ∫ₚ(v⊙(dconv∘(du,∇(du),u,∇(u))),dΩ)

  res(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) + c(u,v) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) + dc(u,du,v) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  # test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
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
  norm_style = [:l2,:l2]
  compute_supremizers = true
  nsnaps_state = 50
  nsnaps_mdeim = 30
  nsnaps_test = 10
  st_mdeim = false
  postprocess = true
  info = RBInfo(test_path;ϵ,norm_style,compute_supremizers,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,st_mdeim,postprocess)
end

# Offline phase
printstyled("OFFLINE PHASE\n";bold=true,underline=true)
if load_solutions
  sols,params = load(info,(BlockSnapshots,Table))
else
  params = realization(feop,nsnaps_state+nsnaps_test)
  sols,stats = collect_multi_field_solutions(fesolver,feop,params)
  if save_solutions
    save(info,(sols,params,stats))
  end
end
if load_structures
  rbspace = load(info,BlockRBSpace)
  rbrhs,rblhs = load(info,(BlockRBVecAlgebraicContribution,Vector{BlockRBMatAlgebraicContribution}))
else
  rbspace = reduced_basis(info,feop,sols)
  rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,params)
  if save_structures
    save(info,(rbspace,rbrhs,rblhs))
  end
end

# Online phase
# multi_field_rb_solver(info,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)

nsnaps_test = info.nsnaps_test
ntimes = get_time_ndofs(fesolver)
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1:1]
x = nearest_neighbor(sols,params,μn)
op = get_ptoperator(fesolver,feop,x,μn)
xrb = space_time_projection(x,op,rbspace)
rhs_cache,lhs_cache = allocate_cache(op,x)
_,conv0 = Algebra._check_convergence(fesolver.nls.ls,xrb)
for iter in 1:fesolver.nls.max_nliters
  x .= recenter(x,fesolver.uh0(μn);θ=fesolver.θ)
  xrb = space_time_projection(x,op,rbspace)
  lrhs = collect_rhs_contributions!(rhs_cache,info,op,rbrhs,rbspace)
  llhs = collect_lhs_contributions!(lhs_cache,info,op,rblhs,rbspace)
  nlrhs = collect_rhs_contributions!(rhs_cache,info,op,nl_rbrhs,rbspace)
  nllhs = collect_lhs_contributions!(lhs_cache,info,op,nl_rblhs,rbspace)
  lhs = llhs + nllhs
  rhs = llhs*xrb + (lrhs+nlrhs)
  xrb = PTArray([lhs[1] \ rhs[1]])
  x -= vcat(recast(xrb,rbspace)...)
  op = update_ptoperator(op,x)
  isconv,conv = Algebra._check_convergence(fesolver.nls,xrb,conv0)
  println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
  if all(isconv); return; end
  if iter == nls.max_nliters
    @unreachable
  end
end
