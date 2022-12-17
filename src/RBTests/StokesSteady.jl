include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function stokes_steady()
  run_fem = false

  steady = true
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/stokes"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(25:26),"neumann" => collect(1:24))
  order = 2

  ranges = fill([1.,2.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,b,bfe,f,ffe,h,hfe,g,lhs,rhs = stokes_functions(ptype,measures)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{3,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1;space=:P)
  V = MyTests(model,reffe1;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)
  Q = MyTests(model,reffe2;conformity=:L2)
  P = MyTrials(Q)
  Y = ParamMultiFieldFESpace([V,Q])
  X = ParamMultiFieldFESpace([U,P])

  op = ParamAffineFEOperator(lhs,rhs,PS,X,Y)

  solver = LinearFESolver()
  nsnap = 100
  uh,ph,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap)

  opA = NonaffineParamVarOperator(a,afe,PS,U,V;id=:A)
  opB = AffineParamVarOperator(b,bfe,PS,U,Q;id=:B)
  opF = AffineParamVarOperator(f,ffe,PS,V;id=:F)
  opH = AffineParamVarOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false)
  tt = TimeTracker(0.,0.)
  rbspace,varinfo = offline_phase(info,(uh,ph,μ),(opA,opB,opF,opH),measures,tt)
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
  opA,opB,opF,opH = op

  rbspace_u,rbspace_p = rb(info,tt,(uh_offline,ph_offline),opB,ph,μ)

  rbopA = RBVarOperator(opA,rbspace_u,rbspace_u)
  rbopB = RBVarOperator(opB,rbspace_p,rbspace_u)
  rbopF = RBVarOperator(opF,rbspace_u)
  rbopH = RBVarOperator(opH,rbspace_u)

  if info.load_offline
    A_rb = load_rb_structure(info,rbopA,meas.dΩ)
    B_rb = load_rb_structure(info,rbopB,meas.dΩ)
    F_rb = load_rb_structure(info,rbopF,meas.dΩ)
    H_rb = load_rb_structure(info,rbopH,meas.dΓn)
  else
    A_rb = assemble_rb_structure(info,tt,rbopA,μ,meas,:dΩ)
    B_rb = assemble_rb_structure(info,tt,rbopB,μ,meas,:dΩ)
    F_rb = assemble_rb_structure(info,tt,rbopF,μ,meas,:dΩ)
    H_rb = assemble_rb_structure(info,tt,rbopH,μ,meas,:dΓn)
  end

  rbspace = (rbspace_u,rbspace_p)
  varinfo = ((rbopA,A_rb),(rbopB,B_rb),(rbopF,F_rb),(rbopH,H_rb))
  rbspace,varinfo
end

function online_phase(
  info::RBInfo,
  fe_sol,
  rbspace::NTuple{2,RBSpace},
  varinfo::Tuple,
  tt::TimeTracker)

  uh,ph,μ = fe_sol

  Ainfo,Binfo,Finfo,Hinfo = varinfo
  rbopA,A_rb = Ainfo
  rbopB,B_rb = Binfo
  rbopF,F_rb = Finfo
  rbopH,H_rb = Hinfo

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      Aon = online_assembler(rbopA,A_rb,μ[k])
      Bon = online_assembler(rbopB,B_rb,μ[k])
      Fon = online_assembler(rbopF,F_rb,μ[k])
      Hon = online_assembler(rbopH,H_rb,μ[k])
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
