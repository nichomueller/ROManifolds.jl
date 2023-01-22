include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function stokes_steady()
  run_fem = true

  steady = true
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/stokes"
  mesh = "cylinder.json"
  bnd_info = Dict("dirichlet" => ["wall","inlet"],"neumann" => ["outlet"])
  order = 2

  ranges = fill([1.,5.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,b,bfe,f,ffe,h,hfe,g,lhs,rhs = stokes_functions(ptype,measures)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  Q = MyTests(model,reffe2;conformity=:C0)
  P = MyTrials(Q)
  Y = ParamMultiFieldFESpace([V,Q])
  X = ParamMultiFieldFESpace([U,P])

  op = ParamAffineFEOperator(lhs,rhs,PS,X,Y)

  solver = LinearFESolver()
  nsnap = 1
  uh,ph,μ, = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap)

  opA = NonaffineParamOperator(a,afe,PS,U,V;id=:A)
  opB = AffineParamOperator(b,bfe,PS,U,Q;id=:B)
  opF = AffineParamOperator(f,ffe,PS,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=30,load_offline=false)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rbspace,rb_structures = offline_phase(info,(uh,ph,μ),(opA,opB,opF,opH),measures,tt)
  online_phase(info,(uh,ph,μ),rbspace,rb_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::NTuple{N,ParamOperator},
  meas::ProblemMeasures,
  tt::TimeTracker) where N

  uh,ph,μ = fesol
  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]
  opA,opB,opF,opH = op

  rbspace_u,rbspace_p = rb(info,tt,(uh_offline,ph_offline),opB,ph,μ)

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)

  Arb = RBStructure(info,tt,rbopA,μ,meas,:dΩ)
  Brb = RBStructure(info,tt,rbopB,μ,meas,:dΩ)
  Frb = RBStructure(info,tt,rbopF,μ,meas,:dΩ)
  Hrb = RBStructure(info,tt,rbopH,μ,meas,:dΓn)

  rbspace = (rbspace_u,rbspace_p)
  rb_structures = ((rbopA,Arb),(rbopB,Brb),(rbopF,Frb),(rbopH,Hrb))
  rbspace,rb_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::NTuple{2,RBSpace},
  rb_structures::Tuple,
  tt::TimeTracker)

  uh,ph,μ = fesol

  Arb,Brb,Frb,Hrb = rb_structures
  rbopA,Arb = Arb
  rbopB,Brb = Brb
  rbopF,Frb = Frb
  rbopH,Hrb = Hrb

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      Aon = online_assembler(rbopA,Arb,μ[k])
      Bon = online_assembler(rbopB,Brb,μ[k])
      Fon = online_assembler(rbopF,Frb,μ[k])
      Hon = online_assembler(rbopH,Hrb,μ[k])
      lift = Aon[2],Bon[2]
      sys = stokes_rb_system((Aon[1],Bon[1]),(Fon,Hon,lift...))
      rb_sol = solve_rb_system(sys...)
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

stokes_steady()
