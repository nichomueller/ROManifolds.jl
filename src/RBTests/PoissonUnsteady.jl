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

  t0,tF,dt,θ = 0.,0.5,0.025,0.5
  time_info = TimeInfo(t0,tF,dt,θ)

  ranges = fill([1.,10.],6)
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

  opA = NonaffineParamVarOperator(a,afe,PS,U,V;id=:A)
  opM = AffineParamVarOperator(m,mfe,PS,U,V;id=:M)
  opF = AffineParamVarOperator(f,ffe,PS,V;id=:F)
  opH = NonaffineParamVarOperator(h,hfe,PS,V;id=:H)

  info = RBInfoUnsteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false)

  tt,rbspace,Ainfo,Minfo,Finfo,Hinfo = offline_phase(info,time_info,[uh,μ],[opA,opM,opF,opH],measures)
  online_phase(info,time_info,[uh,μ],rbspace,[Ainfo,Minfo,Finfo,Hinfo],tt)
end

function offline_phase(
  info::RBInfo,
  time_info::TimeInfo,
  fe_sol,
  op::Vector{<:ParamVarOperator},
  meas::ProblemMeasures)

  uh,μ = fe_sol
  opA,opM,opF,opH = op
  tt = TimeTracker(0.,0.)

  rbspace = rb(info,tt,uh)
  rbopA = RBVarOperator(opA,rbspace,rbspace)
  rbopM = RBVarOperator(opM,rbspace,rbspace)
  rbopF = RBVarOperator(opF,rbspace)
  rbopH = RBVarOperator(opH,rbspace)

  if info.load_offline
    A_rb = load_rb_structure(info,rbopA,meas.dΩ)
    M_rb = load_rb_structure(info,rbopM,meas.dΩ)
    F_rb = load_rb_structure(info,rbopF,meas.dΩ)
    H_rb = load_rb_structure(info,rbopH,meas.dΓn)
  else
    A_rb = assemble_rb_structure(info,tt,rbopA,μ,meas,:dΩ,time_info)
    M_rb = assemble_rb_structure(info,tt,rbopM,μ,meas,:dΩ,time_info)
    F_rb = assemble_rb_structure(info,tt,rbopF,μ,meas,:dΩ,time_info)
    H_rb = assemble_rb_structure(info,tt,rbopH,μ,meas,:dΓn,time_info)
  end

  tt,rbspace,(rbopA,A_rb),(rbopM,M_rb),(rbopF,F_rb),(rbopH,H_rb)
end

function online_phase(
  info::RBInfo,
  time_info::TimeInfo,
  fe_sol,
  rbspace::RBSpace,
  varinfo::Vector{<:Tuple},
  tt::TimeTracker)

  uh,μ = fe_sol
  Nt = get_Nt(uh)

  Ainfo,Minfo,Finfo,Hinfo = varinfo
  rbopA,A_rb = Ainfo
  rbopM,M_rb = Minfo
  rbopF,F_rb = Finfo
  rbopH,H_rb = Hinfo

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      lhs = assemble_rb_system(rbopA,A_rb,μ[k],time_info),assemble_rb_system(rbopM,M_rb,μ[k],time_info)
      rhs = assemble_rb_system(rbopF,F_rb,μ[k],time_info),assemble_rb_system(rbopH,H_rb,μ[k],time_info)
      sys = poisson_rb_system([lhs[1],lhs[3]],[rhs...,lhs[2]],get_θ(time_info))
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
