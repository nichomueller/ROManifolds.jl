include("../FEM/FEM.jl")
include("../RB/RB.jl")
include("tests.jl")

function poisson_steady()
  steady = true
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)
  run_fem = false

  root = "/home/nicholasmueller/git_repos/Mabla.jl/tests/poisson"
  mesh = "cube5x5x5.json"
  bnd_info = Dict("dirichlet" => collect(1:25),"neumann" => [26])
  order = 1
  degree = get_degree(order)

  ranges = fill([1.,10.],6)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(ptype,mesh,root)
  model = model_info(bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,degree)
  V = MyTests(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = MyTrials(V,g,ptype)

  op = ParamAffineFEOperator(lhs,rhs,PS,get_trial(U),get_test(V))

  solver = LinearFESolver()
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,100)

  opA = NonaffineParamVarOperator(a,afe,PS,U,V;id=:A)
  opF = AffineParamVarOperator(f,ffe,PS,V;id=:F)
  opH = NonaffineParamVarOperator(h,hfe,PS,V;id=:H)

  info = RBInfoSteady(ptype,mesh,root;ϵ=1e-5,nsnap=80,mdeim_snap=20,load_offline=false)

  tt,rbspace,rbA,rbF,rbH = offline_phase(info,[uh,μ],[opA,opF,opH],measures)
  online_phase(info,[uh,μ],rbspace,tt,rbA,rbF,rbH)
end

function offline_phase(
  info::RBInfo,
  fe_sol::Vector{Snapshots},
  op::Vector{<:RBVarOperator},
  meas::ProblemMeasures)

  uh,μ = fe_sol
  opA,opF,opH = op
  tt = TimeTracker(0.,0.)

  if info.load_offline
    rbspace = get_rb(info)
    rbA = load_rb_structure(info,opA,rbspace,rbspace,tt,μ,meas,:dΩ)
    rbF = load_rb_structure(info,opF)
    rbH = load_rb_structure(info,opH,rbspace,tt,μ,meas,:dΓn)
    tt,rbspace,rbA,rbF,rbH
  else
    rbspace = assemble_rb(info,tt,uh)
    rbopA = RBVarOperator(opA,rbspace,rbspace)
    rbopF = RBVarOperator(opF,rbspace)
    rbopH = RBVarOperator(opH,rbspace)
    rbA = assemble_rb_structures(info,tt,rbopA,μ,meas,:dΩ)
    rbF = assemble_rb_structures(info,tt,rbopF)
    rbH = assemble_rb_structures(info,tt,rbopH,μ,meas,:dΓn)
    tt,rbspace,rbA,rbF,rbH
  end
end

function online_phase(
  info::RBInfo,
  fe_sol::Vector{Snapshots},
  rbspace::RBSpace,
  op::Vector{<:RBVarOperator},
  tt::TimeTracker,
  args...)

  uh,μ = fe_sol
  rbopA,rbopF,rbopH = op
  rbA,rbF,rbH = args

  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      get_parameter(rbopA,μ[k],rbA)
      get_parameter(rbopF,μ[k],rbF)
      get_parameter(rbopH,μ[k],rbH)
      lhs = assemble_rb_system(rbopA,rbA,μ[k])
      rhs = assemble_rb_system(rbopF,rbF,μ[k]),assemble_rb_system(rbopH,rbH,μ[k])
      sys = poisson_rb_system(lhs,rhs)
      rb_sol = solve_rb_system(sys...)
    end
    uh_rb = reconstruct_fe_sol(rbspace,rb_sol)
    ErrorTracker(:u,uh,uh_rb,k)
  end

  ets = online_loop.(info.online_snaps)
  res = RBResults(:u,tt,ets)
  save(info,res)

  if info.postprocess
    postprocess(info,res)
  end
end
