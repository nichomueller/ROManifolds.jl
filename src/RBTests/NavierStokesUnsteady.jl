include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function navier_stokes_unsteady()
  run_fem = false

  steady = false
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/navier-stokes"
  mesh = "cylinder_h03.json"#"cylinder.json"
  bnd_info = Dict("dirichlet" => ["wall","inlet","inlet_c","inlet_p","outlet_c","outlet_p"],
                  "neumann" => ["outlet"])
  order = 2

  t0,tF,dt,θ = 0.,2,0.05,1#0.,5,0.05,1
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,2.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,m,mfe,b,bfe,bTfe,c,cfe,d,dfe,f,ffe,h,hfe,g,res,jac,jac_t =
    navier_stokes_functions(ptype,measures)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  Q = MyTests(model,reffe2;conformity=:C0)
  P = MyTrials(Q)
  Y = ParamTransientMultiFieldFESpace([V,Q])
  X = ParamTransientMultiFieldFESpace([U,P])

  op = ParamTransientFEOperator(res,jac,jac_t,PS,X,Y)

  nls = NLSolver(show_trace=true,method=:newton,linesearch=BackTracking())
  solver = ThetaMethod(nls,dt,θ)
  nsnap = 100
  uh,ph,μ,ghθ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF;get_lift=true)

  opA = NonaffineParamOperator(a,afe,PS,time_info,U,V;id=:A)
  opB = AffineParamOperator(b,bfe,PS,time_info,U,Q;id=:B)
  opBT = AffineParamOperator(b,bTfe,PS,time_info,P,V;id=:BT)
  opC = NonlinearParamOperator(c,cfe,PS,time_info,U,V;id=:C)
  opD = NonlinearParamOperator(d,dfe,PS,time_info,U,V;id=:D)
  opM = AffineParamOperator(m,mfe,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(f,ffe,PS,time_info,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,time_info,V;id=:H)

  varop = (opA,opB,opBT,opC,opD,opM,opF,opH)
  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,online_snaps=95:100,mdeim_snap=20,load_offline=true)
  fesol = (uh,ph,ghθ,μ,time_info)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rbspace,rbspaceθ,param_on_structures = offline_phase(info,fesol,varop,measures,tt);
  online_phase(info,fesol,rbspace,rbspaceθ,param_on_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::NTuple{N,ParamOperator},
  meas::ProblemMeasures,
  tt::TimeTracker) where N

  println("\n Offline phase, reduced basis method")

  uh,ph,ghθ,μ,_ = fesol
  opA,opB,opBT,opC,opD,opM,opF,opH = op
  θ = get_θ(opA)
  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]
  ghθ_offline = ghθ[1:info.nsnap]
  uhθ_offline = compute_in_timesθ(uh_offline,θ)
  phθ_offline = compute_in_timesθ(ph_offline,θ)

  rbspace_u,rbspace_p = rb(info,tt,(uh_offline,ph_offline),opB,ph,μ)
  rbspace_uθ, = rb(info,tt,(uhθ_offline,phθ_offline),opB,ph,μ)
  rbspace_gθ = rb(info,tt,ghθ_offline;sparsity=true)

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopBT = RBVariable(opBT,rbspace_u,rbspace_p)
  rbopC = RBVariable(opC,rbspace_u,rbspace_u)
  rbopD = RBVariable(opD,rbspace_u,rbspace_u)
  rbopM = RBVariable(opM,rbspace_u,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)

  rbspace = (rbspace_u,rbspace_p)
  rbspaceθ = (rbspace_uθ,rbspace_gθ)

  Arb = RBOfflineStructure(info,tt,rbopA,μ,meas,:dΩ)
  Brb = RBOfflineStructure(info,tt,rbopB,μ,meas,:dΩ)
  BTrb = RBOfflineStructure(info,tt,rbopBT,μ,meas,:dΩ)
  Crb = RBOfflineStructure(info,tt,rbopC,μ,meas,:dΩ,rbspaceθ)
  Drb = RBOfflineStructure(info,tt,rbopD,μ,meas,:dΩ,rbspaceθ)
  Mrb = RBOfflineStructure(info,tt,rbopM,μ,meas,:dΩ)
  Frb = RBOfflineStructure(info,tt,rbopF,μ,meas,:dΩ)
  Hrb = RBOfflineStructure(info,tt,rbopH,μ,meas,:dΓn)

  Arb_eval = eval_off_structure(Arb)
  Brb_eval = eval_off_structure(Brb)
  BTrb_eval = eval_off_structure(BTrb)
  Crb_eval = eval_off_structure(Crb,rbspaceθ)
  Drb_eval = eval_off_structure(Drb,rbspaceθ)
  Mrb_eval = eval_off_structure(Mrb)
  Frb_eval = eval_off_structure(Frb)
  Hrb_eval = eval_off_structure(Hrb)

  Aon_param = RBParamOnlineStructure(Arb,Arb_eval;st_mdeim=info.st_mdeim)
  Bon_param = RBParamOnlineStructure(Brb,Brb_eval;st_mdeim=info.st_mdeim)
  BTon_param = RBParamOnlineStructure(BTrb,BTrb_eval;st_mdeim=info.st_mdeim)
  Con_param = RBParamOnlineStructure(Crb,Crb_eval;st_mdeim=info.st_mdeim)
  Don_param = RBParamOnlineStructure(Drb,Drb_eval;st_mdeim=info.st_mdeim)
  Mon_param = RBParamOnlineStructure(Mrb,Mrb_eval;st_mdeim=info.st_mdeim)
  Fon_param = RBParamOnlineStructure(Frb,Frb_eval;st_mdeim=info.st_mdeim)
  Hon_param = RBParamOnlineStructure(Hrb,Hrb_eval;st_mdeim=info.st_mdeim)

  param_on_structures = Aon_param,Bon_param,BTon_param,Con_param,Don_param,Mon_param,Fon_param,Hon_param

  rbspace,rbspaceθ,param_on_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::NTuple{2,RBSpace},
  rbspaceθ::NTuple{2,RBSpace},
  param_on_structures::Tuple,
  tt::TimeTracker)

  println("\n Online phase, reduced basis method")

  uh,ph,ghθ,μ,time_info = fesol
  timesθ = get_timesθ(time_info)
  θ = get_θ(time_info)
  μ_offline = μ[1:info.nsnap]

  rb_solver(res,jac,x0,ud) = solve_rb_system(res,jac,x0,ud,rbspace,rbspaceθ,timesθ,θ)

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      res,jac = unsteady_navier_stokes_rb_system(expand(param_on_structures),μ[k])
      x0 = initial_guess(rbspace,uh,ph,μ_offline,μ[k])
      rb_sol = rb_solver(res,jac,x0,get_snap(ghθ[k])[:])
    end
    uhk,phk = get_snap(uh[k]),get_snap(ph[k])
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

navier_stokes_unsteady()
