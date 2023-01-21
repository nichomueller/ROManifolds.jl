include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function navier_stokes_steady()
  run_fem = false

  steady = true
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes"
  mesh = "cylinder.json"
  bnd_info = Dict("dirichlet" => ["wall","inlet"],"neumann" => ["outlet"])
  order = 2

  ranges = fill([1.,2.],9)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,b,bfe,c,cfe,d,dfe,f,ffe,h,hfe,g,res,jac = navier_stokes_functions(ptype,measures)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  Q = MyTests(model,reffe2;conformity=:C0)
  P = MyTrials(Q)
  Y = ParamMultiFieldFESpace([V,Q])
  X = ParamMultiFieldFESpace([U,P])

  op = ParamFEOperator(res,jac,PS,X,Y)

  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = FESolver(nls)
  nsnap = 1
  uh,ph,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap)

  opA = NonaffineParamOperator(a,afe,PS,U,V;id=:A)
  opB = AffineParamOperator(b,bfe,PS,U,Q;id=:B)
  opC = NonlinearParamOperator(c,cfe,PS,U,V;id=:C)
  opD = NonlinearParamOperator(d,dfe,PS,U,V;id=:D)
  opF = AffineParamOperator(f,ffe,PS,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=30,load_offline=true)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rbspace,rb_structures = offline_phase(info,(uh,ph,μ,X,Y),(opA,opB,opC,opD,opF,opH),measures,tt)
  online_phase(info,(uh,ph,μ,X,Y),rbspace,rb_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::NTuple{N,ParamOperator},
  meas::ProblemMeasures,
  tt::TimeTracker) where N

  uh,ph,μ, = fesol
  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]
  opA,opB,opC,opD,opF,opH = op

  rbspace_u,rbspace_p = rb(info,tt,(uh_offline,ph_offline),opB,ph,μ)

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopC = RBVariable(opC,rbspace_u,rbspace_u)
  rbopD = RBVariable(opD,rbspace_u,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)

  Arb = RBStructure(info,tt,rbopA,μ,meas,:dΩ)
  Brb = RBStructure(info,tt,rbopB,μ,meas,:dΩ)
  Crb = RBStructure(info,tt,rbopC,μ,meas,:dΩ)
  Drb = RBStructure(info,tt,rbopD,μ,meas,:dΩ)
  Frb = RBStructure(info,tt,rbopF,μ,meas,:dΩ)
  Hrb = RBStructure(info,tt,rbopH,μ,meas,:dΓn)

  rbspace = (rbspace_u,rbspace_p)
  rb_structures = Arb,Brb,Crb,Drb,Frb,Hrb
  rbspace,rb_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::NTuple{2,RBSpace},
  rb_structures::Tuple,
  tt::TimeTracker)

  uh,ph,μ,X,Y = fesol

  Arb,Brb,Crb,Drb,Frb,Hrb = rb_structures

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      Aon = online_assembler(Arb,μ[k])
      Bon = online_assembler(Brb,μ[k])
      Con = online_assembler(Crb,μ[k])
      Don = online_assembler(Drb,μ[k])
      Fon = online_assembler(Frb,μ[k])
      Hon = online_assembler(Hrb,μ[k])
      lift = Aon[2],Bon[2],Con[2]
      sys = navier_stokes_rb_system((Aon[1],Bon[1],Con[1],Don[1]),(Fon,Hon,lift...))
      Uk = X[1](μ[k])
      Vk = Y[1]
      rb_sol = solve_rb_system(sys...,(Uk,Vk),rbspace)
    end
    uhk = get_snap(uh[k])
    phk = get_snap(ph[k])
    uhk_rb,phk_rb = reconstruct_fe_sol(rbspace,rb_sol)
    ErrorTracker(:u,uhk,uhk_rb,k),ErrorTracker(:p,phk,phk_rb,k)
  end

  ets = online_loop.(info.online_snaps)
  ets_u,ets_p = first.(ets),last.(ets)
  res_u,res_p = RBResults(:u,tt,ets_u),RBResults(:p,tt,ets_p)
  save(info,res_u)
  save(info,res_p)

  if info.postprocess
    postprocess(info,(res_u,res_p))
  end
end

navier_stokes_steady()
