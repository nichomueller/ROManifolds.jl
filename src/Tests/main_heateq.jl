begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

function heat_equation()
  root = pwd()
  # mesh = "elasticity_3cyl2D.json"
  # bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
  mesh = "cube2x2.json"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  test_path = "$root/tests/poisson/unsteady/$mesh"
  order = 1
  degree = 2
  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  f(x,μ,t) = 1.
  f(μ,t) = x->f(x,μ,t)
  fμt(μ,t) = PTFunction(f,μ,t)

  h(x,μ,t) = abs(cos(t/μ[3]))
  h(μ,t) = x->h(x,μ,t)
  hμt(μ,t) = PTFunction(h,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = 0
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)

  res(μ,t,u,v,(dΩ,dΓn)) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
  res(μ,t,u,v) = res(μ,t,u,v,(dΩ,dΓn))
  jac(μ,t,u,du,v,(dΩ,)) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
  jac(μ,t,u,du,v) = jac(μ,t,u,du,v,(dΩ,))
  jac_t(μ,t,u,dut,v,(dΩ,)) = ∫(v*dut)dΩ
  jac_t(μ,t,u,du,v) = jac_t(μ,t,u,du,v,(dΩ,))

  T = Float
  reffe = ReferenceFE(lagrangian,T,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = PTTrialFESpace(test,g)
  feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.3,0.005,0.5
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = true
  save_solutions = true
  load_structures = false
  save_structures = true
  norm_style = :l2
  nsnaps_state = 50
  nsnaps_mdeim = 20
  nsnaps_test = 10
  st_mdeim = false
  rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

  # Offline phase
  printstyled("OFFLINE PHASE\n";bold=true,underline=true)
  if load_solutions
    sols,params = load(rbinfo,(Snapshots{Vector{T}},Table))
  else
    params = realization(feop,nsnaps_state+nsnaps_test)
    sols,stats = collect_single_field_solutions(fesolver,feop,params)
    if save_solutions
      save(rbinfo,(sols,params,stats))
    end
  end
  if load_structures
    rbspace = load(rbinfo,RBSpace{T})
    rbrhs,rblhs = load(rbinfo,(RBVecAlgebraicContribution{T},
      Vector{RBMatAlgebraicContribution{T}}))
  else
    rbspace = reduced_basis(rbinfo,feop,sols)
    rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
    if save_structures
      save(rbinfo,(rbspace,rbrhs,rblhs))
    end
  end
  # Online phase
  printstyled("ONLINE PHASE\n";bold=true,underline=true)
  rb_solver(rbinfo,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)
end

heat_equation()

nsnaps_test = rbinfo.nsnaps_test
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)
rhs_cache,lhs_cache = allocate_cache(op,snaps_test)
mdeim_cache,rb_cache = rhs_cache
st_mdeim = rbinfo.st_mdeim
k = RBVecContributionMap(Float)
collect_cache,coeff_cache = mdeim_cache
_res = collect_reduced_residuals!(collect_cache,op,rbrhs)
rb_res_contribs = Vector{PTArray{Vector{Float}}}(undef,length(rbrhs))
for (rbresi,resi) in zip(rbrhs,_res)
  coeff = rb_coefficient!(coeff_cache,rbresi,resi;st_mdeim)
  rb_res_contribs[i] = rb_contribution!(rb_cache,k,rbresi,coeff)
end
