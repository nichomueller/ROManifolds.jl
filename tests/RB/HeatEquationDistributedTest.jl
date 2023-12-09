function main_poisson(;nprocs,ncells=(20,20),options=OPTIONS_CG_JACOBI)
  with_mpi() do distribute
    main_poisson(distribute,nprocs,ncells,options)
  end
end

function main_poisson(distribute,nprocs,ncells,options)
  root = pwd()
  test_path = "$root/results/HeatEquation/cube_$ncell_$ncell"
  ϵ = 1e-4
  load_solutions = true
  save_solutions = true
  load_structures = false
  save_structures = true
  postprocess = true
  norm_style = :H1
  nsnaps_state = 50
  nsnaps_mdeim = 20
  nsnaps_test = 10
  st_mdeim = true
  rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

  ranks = distribute(LinearIndices((prod(nprocs),)))

  GridapPETSc.with(args=split(options)) do
    domain = (0,1,0,1)
    model = CartesianDiscreteModel(ranks,nprocs,domain,ncells)

    order = 1
    degree = 2*order
    Ω = Triangulation(model)
    Γn = BoundaryTriangulation(model,tags=["neumann"])
    dΩ = Measure(Ω,degree)
    dΓn = Measure(Γn,degree)

    ranges = fill([1.,10.],3)
    pspace = PSpace(ranges)

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

    res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
    jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
    jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

    T = Float
    reffe = ReferenceFE(lagrangian,T,order)
    test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
    trial = PTTrialFESpace(test,g)
    feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
    t0,tf,dt,θ = 0.,0.3,0.005,0.5
    uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
    fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

    sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
    rbspace = reduced_basis(rbinfo,feop,sols)
    rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
  end
end
