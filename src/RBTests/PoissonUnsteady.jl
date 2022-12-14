include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("RBTests.jl")

function poisson_unsteady()
  run_fem = false

  steady = false
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  order = 1

  t0,tF,dt,θ = 0.,2.5,0.05,0.5
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,2.],9)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  mshpath = mesh_path(mesh,root)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,m,mfe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)

  op = ParamTransientAffineFEOperator(mfe,lhs,rhs,PS,get_trial(U),get_test(V))

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)

  opA = NonaffineParamVarOperator(a,afe,PS,time_info,U,V;id=:A)
  opM = AffineParamVarOperator(m,mfe,PS,time_info,U,V;id=:M)
  #opF = AffineParamVarOperator(f,ffe,PS,time_info,V;id=:F)
  opF = NonaffineParamVarOperator(f,ffe,PS,time_info,V;id=:F)
  opH = NonaffineParamVarOperator(h,hfe,PS,time_info,V;id=:H)

  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false)

  tt,rbspace,varinfo = offline_phase(info,[uh,μ],[opA,opM,opF,opH],measures)
  online_phase(info,[uh,μ],rbspace,[varinfo...],tt)
end

function offline_phase(
  info::RBInfo,
  fe_sol,
  op::Vector{<:ParamVarOperator},
  meas::ProblemMeasures)

  uh,μ = fe_sol
  uh_offline = uh[1:info.nsnap]
  opA,opM,opF,opH = op
  tt = TimeTracker(0.,0.)

  rbspace = rb(info,tt,uh_offline)
  rbopA = RBVarOperator(opA,rbspace,rbspace)
  rbopA_lift = RBLiftingOperator(rbopA)
  rbopM = RBVarOperator(opM,rbspace,rbspace)
  rbopM_lift = RBLiftingOperator(rbopM)
  rbopF = RBVarOperator(opF,rbspace)
  rbopH = RBVarOperator(opH,rbspace)

  if info.load_offline
    A_rb = load_rb_structure(info,rbopA,get_dΩ(meas))
    A_rb_lift = load_rb_structure(info,rbopA_lift,get_dΩ(meas))
    M_rb = load_rb_structure(info,rbopM,get_dΩ(meas))
    M_rb_lift = load_rb_structure(info,rbopM_lift,get_dΩ(meas))
    F_rb = load_rb_structure(info,rbopF,get_dΩ(meas))
    H_rb = load_rb_structure(info,rbopH,get_dΓn(meas))
  else
    A_rb,A_rb_lift = assemble_rb_structure(info,tt,rbopA,μ,meas,:dΩ)
    M_rb,M_rb_lift = assemble_rb_structure(info,tt,rbopM,μ,meas,:dΩ)
    F_rb = assemble_rb_structure(info,tt,rbopF,μ,meas,:dΩ)
    H_rb = assemble_rb_structure(info,tt,rbopH,μ,meas,:dΓn)
  end

  varinfo = ((rbopA,A_rb),(rbopM,M_rb),(rbopF,F_rb),
    (rbopH,H_rb),(rbopA_lift,A_rb_lift),(rbopM_lift,M_rb_lift))
  tt,rbspace,varinfo
end

function online_phase(
  info::RBInfo,
  fe_sol,
  rbspace::RBSpace,
  varinfo::Vector{<:Tuple},
  tt::TimeTracker)

  uh,μ = fe_sol
  Nt = get_Nt(uh)

  Ainfo,Minfo,Finfo,Hinfo,Ainfo_lift,Minfo_lift = varinfo
  rbopA,A_rb = Ainfo
  rbopM,M_rb = Minfo
  rbopF,F_rb = Finfo
  rbopH,H_rb = Hinfo
  rbopA_lift,A_rb_lift = Ainfo_lift
  rbopM_lift,M_rb_lift = Minfo_lift

  θ = get_θ(rbopA)

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      lhs = (assemble_rb_system(rbopA,A_rb,μ[k],info.st_mdeim),
        assemble_rb_system(rbopM,M_rb,μ[k],info.st_mdeim))
      rhs = assemble_rb_system(rbopF,F_rb,μ[k],info.st_mdeim),
        assemble_rb_system(rbopH,H_rb,μ[k],info.st_mdeim)
      lift = assemble_rb_system(rbopA_lift,A_rb_lift,μ[k],info.st_mdeim),
        assemble_rb_system(rbopM_lift,M_rb_lift,μ[k],info.st_mdeim)
      sys = poisson_rb_system([lhs[1]...,lhs[2]...],[rhs...,lift...],θ)
      rb_sol = solve_rb_system(sys...)
    end
    uhk = Matrix(get_snap(uh)[:,(k-1)*Nt+1:k*Nt])
    uhk_rb = reconstruct_fe_sol(rbspace,rb_sol)
    ErrorTracker(:u,uhk,uhk_rb,k)
  end

  ets = online_loop.(info.online_snaps)
  res = RBResults(:u,tt,ets)
  save(info,res)

  if info.postprocess
    postprocess(info,res)
  end
end

poisson_unsteady()
