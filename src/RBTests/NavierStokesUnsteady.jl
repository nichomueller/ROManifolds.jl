root = pwd()

@everywhere include("$root/src/FEM/FEM.jl")
@everywhere include("$root/src/RB/RB.jl")
@everywhere include("$root/src/RBTests/RBTests.jl")

function navier_stokes_unsteady()
  run_fem = false

  steady = false
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  test_path = "$root/tests/navier-stokes/$mesh"
  mesh = "flow_3cyl2D.json"
  bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
  dim = 2
  order = 2

  t0,tF,dt,θ = 0.,0.15,0.0025,1
  time_info = ThetaMethodInfo(t0,tF,dt,θ)

  ranges = [[1.,10.],[0.5,1.],[0.5,1.],[1.,2.]]
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(test_path,mesh)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  function a(x,p::Param,t::Real)
    μ = get_μ(p)
    1e-3*μ[1]
  end
  a(μ::Param,t::Real) = x->a(x,μ,t)
  b(x,μ::Param,t::Real) = 1.
  b(μ::Param,t::Real) = x->b(x,μ,t)
  c(x,μ::Param,t::Real) = 1.
  c(μ::Param,t::Real) = x->c(x,μ,t)
  d(x,μ::Param,t::Real) = 1.
  d(μ::Param,t::Real) = x->d(x,μ,t)
  m(x,μ::Param,t::Real) = 1.
  m(μ::Param,t::Real) = x->m(x,μ,t)
  f(x,p::Param,t::Real) = VectorValue(0.,0.)
  f(μ::Param,t::Real) = x->f(x,μ,t)
  h(x,p::Param,t::Real) = VectorValue(0.,0.)
  h(μ::Param,t::Real) = x->h(x,μ,t)
  function g(x,p::Param,t::Real)
    μ = get_μ(p)
    W = 1.5
    T = 0.16
    flow_rate = μ[4]*abs(1-cos(pi*t/T)+μ[2]*sin(μ[3]*pi*t/T))
    parab_prof = VectorValue(abs.(x[2]*(x[2]-W))/(W/2)^2,0.)
    parab_prof*flow_rate
  end
  g(μ::Param,t::Real) = x->g(x,μ,t)
  g0(x,p::Param,t::Real) = VectorValue(0.,0.)
  g0(p::Param,t::Real) = x->g0(x,p,t)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{dim,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = TestFESpace(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  U = ParamTransientTrialFESpace(V,[g0,g,g])
  Q = TestFESpace(model,reffe2;conformity=:C0)
  P = TrialFESpace(Q)

  feop,opA,opB,opBT,opC,opD,opM,opF,opH =
    navier_stokes_operators(measures,PS,time_info,V,U,Q,P;a,b,c,d,m,f,h)

  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = ThetaMethod(nls,dt,θ)
  nsnap = 100
  uh,ph,μ = fe_snapshots(ptype,solver,feop,fepath,run_fem,nsnap,t0,tF)

  offline_nsnap = 50
  μ_offline = μ[offline_nsnap]
  uh_offline = uh[offline_nsnap]
  ph_offline = ph[offline_nsnap]
  uhθ_offline = compute_in_times(uh_offline,θ)
  ghθ_offline = get_dirichlet_values(U,μ_offline,time_info)
  ughθ_offline = vcat(uhθ_offline,ghθ_offline)

  for fun_mdeim = (true,false), st_mdeim = (true,false), tol = (1e-1,1e-2,1e-3,1e-4)

    global info = RBInfoUnsteady(ptype,test_path;ϵ=tol,nsnap=offline_nsnap,
      online_snaps=95:100,mdeim_snap=15,load_offline=false,postprocess=true,
      fun_mdeim=fun_mdeim,st_mdeim=st_mdeim)
    tt = TimeTracker(OfflineTime(0.,0.),0.)

    printstyled("Offline phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:blue)

    rbspace_u,rbspace_p = rb(info,(uh_offline,ph_offline),opB,ph,μ;tt)
    rbspace = rbspace_u,rbspace_p

    rbopA = RBVariable(opA,rbspace_u,rbspace_u)
    rbopB = RBVariable(opB,rbspace_p,rbspace_u)
    rbopBT = RBVariable(opBT,rbspace_u,rbspace_p)
    rbopC = RBVariable(opC,rbspace_u,rbspace_u)
    rbopD = RBVariable(opD,rbspace_u,rbspace_u)
    rbopM = RBVariable(opM,rbspace_u,rbspace_u)
    rbopF = RBVariable(opF,rbspace_u)
    rbopH = RBVariable(opH,rbspace_u)
    rbopAlift = RBLiftVariable(rbopA)
    rbopBlift = RBLiftVariable(rbopB)
    rbopClift = RBLiftVariable(rbopC)
    rbopMlift = RBLiftVariable(rbopM)

    tt.offline_time.assembly_time = @elapsed begin
      Arb = RBAffineDecomposition(info,rbopA,measures,:dΩ,μ)
      Brb = RBAffineDecomposition(info,rbopB,measures,:dΩ,μ)
      BTrb = RBAffineDecomposition(info,rbopBT,measures,:dΩ,μ)
      Crb = RBAffineDecomposition(info,tt,rbopC,μ,measures,:dΩ,ughθ_offline)
      Drb = RBAffineDecomposition(info,tt,rbopD,μ,measures,:dΩ,ughθ_offline)
      Mrb = RBAffineDecomposition(info,rbopM,measures,:dΩ,μ)
      Frb = RBAffineDecomposition(info,rbopF,measures,:dΩ,μ)
      Hrb = RBAffineDecomposition(info,rbopH,measures,:dΓn,μ)
      Aliftrb = RBAffineDecomposition(info,rbopAlift,measures,:dΩ,μ)
      Bliftrb = RBAffineDecomposition(info,rbopBlift,measures,:dΩ,μ)
      Cliftrb = RBAffineDecomposition(info,rbopClift,measures,:dΩ,μ)
      Mliftrb = RBAffineDecomposition(info,rbopMlift,measures,:dΩ,μ)
    end
    ad = (Arb,Brb,BTrb,Mrb,Frb,Hrb,Aliftrb,Bliftrb,Mliftrb)

    if info.save_offline save(info,(rbspace,ad)) end

    printstyled("Online phase; tol=$tol, st_mdeim=$st_mdeim\n";color=:red)

    Aon = RBParamOnlineStructure(Arb;st_mdeim=info.st_mdeim)
    Bon = RBParamOnlineStructure(Brb;st_mdeim=info.st_mdeim)
    BTon = RBParamOnlineStructure(BTrb;st_mdeim=info.st_mdeim)
    Con = RBParamOnlineStructure(Crb;st_mdeim=info.st_mdeim)
    Don = RBParamOnlineStructure(Drb;st_mdeim=info.st_mdeim)
    Mon = RBParamOnlineStructure(Mrb;st_mdeim=info.st_mdeim)
    Fon = RBParamOnlineStructure(Frb;st_mdeim=info.st_mdeim)
    Hon = RBParamOnlineStructure(Hrb;st_mdeim=info.st_mdeim)
    Alifton = RBParamOnlineStructure(Aliftrb;st_mdeim=info.st_mdeim)
    Blifton = RBParamOnlineStructure(Bliftrb;st_mdeim=info.st_mdeim)
    Clifton = RBParamOnlineStructure(Cliftrb;st_mdeim=info.st_mdeim)
    Mlifton = RBParamOnlineStructure(Mliftrb;st_mdeim=info.st_mdeim)
    online_structures = (Aon,Bon,BTon,Con,Don,Mon,Fon,Hon,Alifton,Blifton,Clifton,Mlifton)

    μ_online = μ[info.online_snaps]
    err_u = ErrorTracker[]
    err_p = ErrorTracker[]
    @distributed for (k,μk) in enumerate(μ_online)
      printstyled("-------------------------------------------------------------\n")
      printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
      tt.online_time += @elapsed begin
        rb_system = unsteady_navier_stokes_rb_system(online_structures,μk)
        x0 = get_initial_guess(uh,ph,μ,μk)
        rb_sol = solve_rb_system(rb_system,x0,rbspace_u,U,μk,time_info;tol=info.ϵ)
      end
      uhk,phk = uh[k],ph[k]
      uhk_rb,phk_rb = reconstruct_fe_sol(rbspace,rb_sol)
      push!(err_u,ErrorTracker(uhk,uhk_rb))
      push!(err_p,ErrorTracker(phk,phk_rb))
    end

    res_u,res_p = RBResults(:u,tt,err_u),RBResults(:p,tt,err_p)
    if info.save_online save(info,(res_u,res_p)) end
    printstyled("Average online wall time: $(tt.online_time/length(err_u)) s\n";
      color=:red)

    if info.postprocess
      trian = get_triangulation(model)
      k = first(info.online_snaps)
      writevtk(info,time_info,uh[k],t->U(μ[k],t),trian)
      writevtk(info,time_info,ph[k],t->P,trian)
      writevtk(info,time_info,res_u,V,trian)
      writevtk(info,time_info,res_p,Q,trian)
    end
  end

  if info.postprocess
    postprocess(info)
  end

end

navier_stokes_unsteady()
