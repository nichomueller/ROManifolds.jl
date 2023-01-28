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
  mesh = "cylinder_h03.json"
  bnd_info = Dict("dirichlet" => ["wall","inlet","inlet_c","inlet_p","outlet_c","outlet_p"],
                  "neumann" => ["outlet"])
  order = 2

  t0,tF,dt,θ = 0.,2,0.05,1
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
  uh,ph,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)

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
  fesol = (uh,ph,μ,time_info,U)
  tt = TimeTracker(OfflineTime(0.,0.),0.)
  rbspace,param_on_structures = offline_phase(info,fesol,varop,measures,tt);
  online_phase(info,fesol,rbspace,param_on_structures,tt)
end

function offline_phase(
  info::RBInfo,
  fesol,
  op::NTuple{N,ParamOperator},
  measures::ProblemMeasures,
  tt::TimeTracker) where N

  println("\n Offline phase, reduced basis method")

  uh,ph,μ,_ = fesol
  opA,opB,opBT,opC,opD,opM,opF,opH = op
  θ = get_θ(opA)
  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]
  uhθ_offline = compute_in_timesθ(uh_offline,θ)

  rbspace_u,rbspace_p = rb(info,tt,(uh_offline,ph_offline),opB,ph,μ)
  rbspace = rbspace_u,rbspace_p

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopBT = RBVariable(opBT,rbspace_u,rbspace_p)
  rbopC = RBVariable(opC,rbspace_u,rbspace_u)
  rbopD = RBVariable(opD,rbspace_u,rbspace_u)
  rbopM = RBVariable(opM,rbspace_u,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)

  Arb = RBAffineDecomposition(info,tt,rbopA,μ,measures,:dΩ)
  Brb = RBAffineDecomposition(info,tt,rbopB,μ,measures,:dΩ)
  BTrb = RBAffineDecomposition(info,tt,rbopBT,μ,measures,:dΩ)
  Crb = RBAffineDecomposition(info,tt,rbopC,μ,measures,:dΩ,uhθ_offline)
  Drb = RBAffineDecomposition(info,tt,rbopD,μ,measures,:dΩ,uhθ_offline)
  Mrb = RBAffineDecomposition(info,tt,rbopM,μ,measures,:dΩ)
  Frb = RBAffineDecomposition(info,tt,rbopF,μ,measures,:dΩ)
  Hrb = RBAffineDecomposition(info,tt,rbopH,μ,measures,:dΓn)

  ad = (Arb,Brb,BTrb,Crb,Drb,Mrb,Frb,Hrb)
  ad_eval = eval_affine_decomposition(ad)
  param_on_structures = RBParamOnlineStructure(ad,ad_eval;st_mdeim=info.st_mdeim)

  rbspace,param_on_structures
end

function online_phase(
  info::RBInfo,
  fesol,
  rbspace::NTuple{2,RBSpace},
  param_on_structures::Tuple,
  tt::TimeTracker)

  println("\n Online phase, reduced basis method")

  uh,ph,μ,time_info,U = fesol
  μ_offline = μ[1:info.nsnap]
  rb_solver(res,jac,x0,Uk) = solve_rb_system(res,jac,x0,Uk,rbspace,time_info)

  function online_loop(k::Int)
    println("Evaluating RB system for μ = μ[$k]")
    tt.online_time += @elapsed begin
      res,jac = unsteady_navier_stokes_rb_system(param_on_structures,μ[k])
      Uk(t) = get_trial(U)(μ[k],t)
      x0 = initial_guess(uh,ph,μ_offline,μ[k])
      rb_sol = rb_solver(res,jac,x0,Uk)
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
