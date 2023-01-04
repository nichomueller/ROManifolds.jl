include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function stokes_unsteady()
  run_fem = false

  steady = false
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/stokes"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  order = 2

  t0,tF,dt,θ = 0.,0.5,0.05,0.5
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,2.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,m,mfe,mfe_gridap,b,bfe,bTfe,f,ffe,h,hfe,g,lhs,rhs = stokes_functions(ptype,measures)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1;space=:P)
  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  Q = MyTests(model,reffe2;conformity=:L2)
  P = MyTrials(Q)
  Y = ParamTransientMultiFieldFESpace([V,Q])
  X = ParamTransientMultiFieldFESpace([U,P])

  op = ParamTransientAffineFEOperator(mfe_gridap,lhs,rhs,PS,X,Y)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
  uh,ph,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)

  opA = NonaffineParamVarOperator(a,afe,PS,time_info,U,V;id=:A)
  opM = AffineParamVarOperator(m,mfe,PS,time_info,U,V;id=:M)
  opB = AffineParamVarOperator(b,bfe,PS,time_info,U,Q;id=:B)
  opBT = AffineParamVarOperator(b,bTfe,PS,time_info,P,V;id=:BT)
  opF = AffineParamVarOperator(f,ffe,PS,time_info,V;id=:F)
  opH = AffineParamVarOperator(h,hfe,PS,time_info,V;id=:H)

  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false,
    st_mdeim=true)
  tt = TimeTracker(0.,0.)
  rbspace,varinfo = offline_phase(info,(uh,ph,μ),(opA,opM,opB,opBT,opF,opH),measures,tt)
  online_phase(info,(uh,ph,μ),rbspace,varinfo,tt)
end

function offline_phase(
  info::RBInfo,
  fe_sol,
  op::NTuple{N,ParamVarOperator},
  meas::ProblemMeasures,
  tt::TimeTracker) where N

  uh,ph,μ = fe_sol
  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]
  opA,opM,opB,opBT,opF,opH = op

  rbspace_u,rbspace_p = rb(info,tt,(uh_offline,ph_offline),opB,ph,μ)

  rbopA = RBVarOperator(opA,rbspace_u,rbspace_u)
  rbopM = RBVarOperator(opM,rbspace_u,rbspace_u)
  rbopB = RBVarOperator(opB,rbspace_p,rbspace_u)
  rbopBT = RBVarOperator(opBT,rbspace_u,rbspace_p)
  rbopF = RBVarOperator(opF,rbspace_u)
  rbopH = RBVarOperator(opH,rbspace_u)

  A_rb = rb_structure(info,tt,rbopA,μ,meas,:dΩ)
  M_rb = rb_structure(info,tt,rbopM,μ,meas,:dΩ)
  B_rb = rb_structure(info,tt,rbopB,μ,meas,:dΩ)
  BT_rb = rb_structure(info,tt,rbopBT,μ,meas,:dΩ)
  F_rb = rb_structure(info,tt,rbopF,μ,meas,:dΩ)
  H_rb = rb_structure(info,tt,rbopH,μ,meas,:dΓn)

  rbspace = (rbspace_u,rbspace_p)
  varinfo = ((rbopA,A_rb),(rbopM,M_rb),(rbopB,B_rb),(rbopBT,BT_rb),(rbopF,F_rb),(rbopH,H_rb))
  rbspace,varinfo
end

function online_phase(
  info::RBInfo,
  fe_sol,
  rbspace::NTuple{2,RBSpace},
  varinfo::Tuple,
  tt::TimeTracker)

  uh,ph,μ = fe_sol

  Ainfo,Minfo,Binfo,BTinfo,Finfo,Hinfo = varinfo
  rbopA,A_rb = Ainfo
  rbopM,M_rb = Minfo
  rbopB,B_rb = Binfo
  rbopBT,BT_rb = BTinfo
  rbopF,F_rb = Finfo
  rbopH,H_rb = Hinfo

  st_mdeim = info.st_mdeim
  θ = get_θ(rbopA)

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      Aon = online_assembler(rbopA,A_rb,μ[k],st_mdeim)
      Mon = online_assembler(rbopM,M_rb,μ[k],st_mdeim)
      Bon = online_assembler(rbopB,B_rb,μ[k],st_mdeim)
      BTon = online_assembler(rbopBT,BT_rb,μ[k],st_mdeim)
      Fon = online_assembler(rbopF,F_rb,μ[k],st_mdeim)
      Hon = online_assembler(rbopH,H_rb,μ[k],st_mdeim)
      lift = Aon[2],Mon[2],Bon[2]
      sys = stokes_rb_system((Aon[1]...,Mon[1]...,BTon...,Bon[1]...),(Fon,Hon,lift...),θ)
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

stokes_unsteady()
