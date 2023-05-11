using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(manager)

@everywhere root = pwd()
@everywhere include("$root/src/FEM/FEM.jl")
@everywhere include("$root/src/RB/RB.jl")
@everywhere include("$root/src/RBTests/RBTests.jl")

@mpi_do manager begin
  run_fem = false

  steady = false
  indef = true
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  mesh = "flow_3cyl_02.json"
  test_path = "$root/tests/stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet_noslip" => ["noslip"],
                  "dirichlet_nopenetration" => ["nopenetration"],
                  "dirichlet" => ["inlet"],
                  "neumann" => ["outlet"])
  order = 2

  fepath = fem_path(test_path)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)
  D = get_dimension(model)

  ranges = fill([1.,10.],2)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  t0,tF,dt,θ = 0.,0.15,0.0025,1
  time_info = ThetaMethodInfo(t0,tF,dt,θ)

  function a(x,p::Param,t::Real)
    exp((cos(t)+sin(t))*x[1]/sum(p.μ))
  end
  a(p::Param,t::Real) = x->a(x,p,t)
  b(x,p::Param,t::Real) = 1.
  b(p::Param,t::Real) = x->b(x,p,t)
  m(x,p::Param,t::Real) = 1.
  m(p::Param,t::Real) = x->m(x,p,t)
  f(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  f(p::Param,t::Real) = x->f(x,p,t)
  h(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  h(p::Param,t::Real) = x->h(x,p,t)
  function g(x,p::Param,t::Real)
    W = 1.5
    T = 0.16
    flow_rate = abs(1-cos(2*pi*t/T)+sin(2*pi*t/(p.μ[1]*T))/p.μ[2])
    parab_prof = VectorValue(abs.(x[2]*(x[2]-W))/(W/2)^2,0.,0.)
    parab_prof*flow_rate
  end
  g(p::Param,t::Real) = x->g(x,p,t)
  g0(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  g0(p::Param,t::Real) = x->g0(x,p,t)

  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{D,Float},order)
  reffe1 = Gridap.ReferenceFE(lagrangian,VectorValue{D,Float},order)
  reffe2 = Gridap.ReferenceFE(lagrangian,Float,order-1)
  V = TestFESpace(model,reffe1;conformity=:H1,
    dirichlet_tags=["dirichlet_noslip","dirichlet_nopenetration","dirichlet"],
    dirichlet_masks=[(true,true,true),(false,false,true),(true,true,true)])
  U = ParamTransientTrialFESpace(V,[g0,g,g])
  Q = TestFESpace(model,reffe2;conformity=:C0)
  P = TrialFESpace(Q)

  feop,opA,opB,opBT,opM,opF,opH = stokes_operators(measures,PS,time_info,V,U,Q,P;a,b,m,f,h)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
  uh,ph,μ = fe_snapshots(solver,feop,fepath,run_fem,nsnap,t0,tF;indef)

  info = RBInfoUnsteady(ptype,test_path;ϵ=1e-4,nsnap=80,mdeim_snap=20,load_offline=false,fun_mdeim=true)
  tt = TimeTracker(OfflineTime(0.,0.),0.)

  printstyled("Offline phase, reduced basis method\n";color=:blue)

  uh_offline = uh[1:info.nsnap]
  ph_offline = ph[1:info.nsnap]

  tt.offline_time.basis_time += @elapsed begin
    rbspace_u,rbspace_p = assemble_rbspace(info,(uh_offline,ph_offline),opB)
  end
  rbspace = rbspace_u,rbspace_p

  rbopA = RBVariable(opA,rbspace_u,rbspace_u)
  rbopB = RBVariable(opB,rbspace_p,rbspace_u)
  rbopBT = RBVariable(opBT,rbspace_u,rbspace_p)
  rbopM = RBVariable(opM,rbspace_u,rbspace_u)
  rbopF = RBVariable(opF,rbspace_u)
  rbopH = RBVariable(opH,rbspace_u)
  rbopAlift = RBLiftVariable(rbopA)
  rbopBlift = RBLiftVariable(rbopB)
  rbopMlift = RBLiftVariable(rbopM)

  tt.offline_time.assembly_time = @elapsed begin
    Arb = RBAffineDecomposition(info,rbopA,measures,:dΩ,μ)
    Brb = RBAffineDecomposition(info,rbopB,measures,:dΩ,μ)
    BTrb = RBAffineDecomposition(info,rbopBT,measures,:dΩ,μ)
    Mrb = RBAffineDecomposition(info,rbopM,measures,:dΩ,μ)
    Frb = RBAffineDecomposition(info,rbopF,measures,:dΩ,μ)
    Hrb = RBAffineDecomposition(info,rbopH,measures,:dΓn,μ)
    Aliftrb = RBAffineDecomposition(info,rbopAlift,measures,:dΩ,μ)
    Bliftrb = RBAffineDecomposition(info,rbopBlift,measures,:dΩ,μ)
    Mliftrb = RBAffineDecomposition(info,rbopMlift,measures,:dΩ,μ)
  end
  ad = (Arb,Brb,BTrb,Mrb,Frb,Hrb,Aliftrb,Bliftrb,Mliftrb)

  if info.save_offline save(info,(rbspace,ad)) end

  printstyled("Online phase, reduced basis method\n";color=:red)

  Aon = RBParamOnlineStructure(Arb;st_mdeim=info.st_mdeim)
  Bon = RBParamOnlineStructure(Brb;st_mdeim=info.st_mdeim)
  BTon = RBParamOnlineStructure(BTrb;st_mdeim=info.st_mdeim)
  Mon = RBParamOnlineStructure(Mrb;st_mdeim=info.st_mdeim)
  Fon = RBParamOnlineStructure(Frb;st_mdeim=info.st_mdeim)
  Hon = RBParamOnlineStructure(Hrb;st_mdeim=info.st_mdeim)
  Alifton = RBParamOnlineStructure(Aliftrb;st_mdeim=info.st_mdeim)
  Blifton = RBParamOnlineStructure(Bliftrb;st_mdeim=info.st_mdeim)
  Mlifton = RBParamOnlineStructure(Mliftrb;st_mdeim=info.st_mdeim)
  online_structures = (Aon,Bon,BTon,Mon,Fon,Hon,Alifton,Blifton,Mlifton)

  err_u = ErrorTracker[]
  err_p = ErrorTracker[]
  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      lhs,rhs = unsteady_stokes_rb_system(online_structures,μ[k])
      rb_sol = solve_rb_system(lhs,rhs)
    end
    uhk,phk = uh[k],ph[k]
    uhk_rb,phk_rb = reconstruct_fe_sol(rbspace,rb_sol)
    push!(err_u,ErrorTracker(uhk,uhk_rb))
    push!(err_p,ErrorTracker(phk,phk_rb))
  end

  pmap(online_loop,info.online_snaps)
  res = RBResults(:u,tt,err_u),RBResults(:p,tt,err_p)
  postprocess(info,res,(V,Q),model,time_info)
end
