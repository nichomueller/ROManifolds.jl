using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(manager)

@everywhere begin
  root = pwd()
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
  include("$root/src/RBTests/RBTests.jl")
end

@everywhere begin
  run_fem = false

  steady = false
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  mesh = "elasticity_3cyl.json"
  test_path = "$root/tests/poisson/unsteady/$mesh"
  bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
  order = 1

  fepath = fem_path(test_path)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  t0,tF,dt,θ = 0.,0.3,0.005,1
  time_info = ThetaMethodInfo(t0,tF,dt,θ)

  a(x,p::Param,t::Real) = exp((sin(t)+cos(t))*x[1]/sum(p.μ))
  a(p::Param,t::Real) = x->a(x,p,t)
  a(p::Param) = t->a(p,t)
  m(x,p::Param,t::Real) = 1.
  m(p::Param,t::Real) = x->m(x,p,t)
  m(p::Param) = t->m(p,t)
  f(x,p::Param,t::Real) = 1.
  f(p::Param,t::Real) = x->f(x,p,t)
  f(p::Param) = t->f(p,t)
  h(x,p::Param,t::Real) = abs(cos(p.μ[3]*t))
  h(p::Param,t::Real) = x->h(x,p,t)
  h(p::Param) = t->h(p,t)
  g(x,p::Param,t::Real) = p.μ[1]*exp(-x[1]/p.μ[2])*abs(sin(p.μ[3]*t))
  g(p::Param,t::Real) = x->g(x,p,t)
  g(p::Param) = t->g(p,t)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = ParamTransientTrialFESpace(V,g)

  feop,opA,opM,opF,opH = poisson_operators(measures,PS,time_info,V,U;a,m,f,h)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
end

@everywhere workers() begin
  u,μ = generate_fe_snapshots_on_workers(run_fem,fepath,nsnap,solver,feop,t0,tF;indef)
end

begin
  uh,μh = collect_fe_snapshots(run_fem,fepath,nsnap;indef)
  @passobj 1 workers() uh
  @passobj 1 workers() μh
end

@mpi_do manager begin
  uh = convert_snapshot(EMatrix{Float},uh)

  info = RBInfoUnsteady(ptype,test_path;ϵ=1e-3,nsnap=80,mdeim_snap=20)

  printstyled("Offline phase, reduced basis method\n";color=:blue)

  uh_offline = uh[1:info.nsnap]

  basis_time = @elapsed begin
    rb_space, = assemble_rb_space(info,(u_offline,))
  end

  rbopA = RBVariable(opA,rb_space,rb_space)
  rbopM = RBVariable(opM,rb_space,rb_space)
  rbopF = RBVariable(opF,rb_space)
  rbopH = RBVariable(opH,rb_space)
  rbopAlift = RBLiftVariable(rbopA)
  rbopMlift = RBLiftVariable(rbopM)

  assembly_time = @elapsed begin
    Arb = RBAffineDecomposition(info,rbopA,measures,:dΩ,μh)
    Mrb = RBAffineDecomposition(info,rbopM,measures,:dΩ,μh)
    Frb = RBAffineDecomposition(info,rbopF,measures,:dΩ,μh)
    Hrb = RBAffineDecomposition(info,rbopH,measures,:dΓn,μh)
    Aliftrb = RBAffineDecomposition(info,rbopAlift,measures,:dΩ,μh)
    Mliftrb = RBAffineDecomposition(info,rbopMlift,measures,:dΩ,μh)
  end
  ad = (Arb,Mrb,Frb,Hrb,Aliftrb,Mliftrb)

  offline_times = OfflineTime(basis_time,assembly_time)

  if info.save_offline
    save(info,(rb_space,ad,offline_times))
  end

  printstyled("Online phase, reduced basis method\n";color=:red)

  Aon = RBParamOnlineStructure(Arb;st_mdeim=info.st_mdeim)
  Mon = RBParamOnlineStructure(Mrb;st_mdeim=info.st_mdeim)
  Fon = RBParamOnlineStructure(Frb;st_mdeim=info.st_mdeim)
  Hon = RBParamOnlineStructure(Hrb;st_mdeim=info.st_mdeim)
  Alifton = RBParamOnlineStructure(Aliftrb;st_mdeim=info.st_mdeim)
  Mlifton = RBParamOnlineStructure(Mliftrb;st_mdeim=info.st_mdeim)
  online_structures = (Aon,Mon,Fon,Hon,Alifton,Mlifton)

  rb_system(k) = unsteady_poisson_rb_system(online_structures,μh[k])
  res = online_loop(k->u[k],rb_space,rb_system,info.online_snaps)
  postprocess(info,(res,),(V,),model,time_info)
end

poisson_unsteady()

#=
run_fem = false

  steady = false
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  test_path = "$root/tests/poisson/$mesh"
  mesh = "model.json"
  bnd_info = Dict("dirichlet" => ["sides","sides_c"],
                  "neumann" => ["circle","triangle","square"])
  order = 1

  t0,tF,dt,θ = 0.,0.3,0.005,0.5
  time_info = ThetaMethodInfo(t0,tF,dt,θ)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  fepath = fem_path(test_path,mesh)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info,ptype)
  measures = ProblemMeasures(model,order)

  a,afe,m,mfe,f,ffe,h,hfe,g,lhs,rhs = poisson_functions(ptype,measures)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = ParamTransientTrialFESpace(V,g)
  op = ParamTransientAffineFEOperator(mfe,lhs,rhs,PS,U,V)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
  u,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)

  opA = NonaffineParamOperator(a,afe,PS,time_info,U,V;id=:A)
  opM = AffineParamOperator(m,mfe,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(f,ffe,PS,time_info,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,time_info,V;id=:H)

for fun_mdeim = (false)#(false,true)
  for st_mdeim = (false)#(false,true)
    for tol = (1e-4)#(1e-2,1e-3,1e-4,1e-5)

      global info = RBInfoUnsteady(ptype,test_path;ϵ=tol,nsnap=80,online_snaps=95:100,
        mdeim_snap=20,save_offline=false,postprocess=true,
        fun_mdeim=fun_mdeim,st_mdeim=st_mdeim,save_online=false)
      tt = TimeTracker(OfflineTime(0.,0.),0.)

      printstyled("Offline phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:blue)

      u_offline = u[1:info.nsnap]
      #X = H1_norm_matrix(opA,opM)
      rb_space = assemble_rb_space(info,tt,u_offline)#;X)

      rbopA = RBVariable(opA,rb_space,rb_space)
      rbopM = RBVariable(opM,rb_space,rb_space)
      rbopF = RBVariable(opF,rb_space)
      rbopH = RBVariable(opH,rb_space)

      Arb = RBAffineDecomposition(info,tt,rbopA,μ,measures,:dΩ)
      Mrb = RBAffineDecomposition(info,tt,rbopM,μ,measures,:dΩ)
      Frb = RBAffineDecomposition(info,tt,rbopF,μ,measures,:dΩ)
      Hrb = RBAffineDecomposition(info,tt,rbopH,μ,measures,:dΓn)

      ad = (Arb,Mrb,Frb,Hrb)
      ad_eval = eval_affine_decomposition(ad)
      online_structures = RBParamOnlineStructure(ad,ad_eval;st_mdeim=info.st_mdeim)

      printstyled("Online phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:red)

      function online_loop(k::Int)
        printstyled("-------------------------------------------------------------\n")
        printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
        tt.online_time += @elapsed begin
          printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
          lhs,rhs = unsteady_poisson_rb_system(online_structures,μ[k])
          rb_sol = solve_rb_system(lhs,rhs)
        end
        uhk = get_snap(u[k])
        uhk_rb = reconstruct_fe_sol(rb_space,rb_sol)
        ErrorTracker(:u,uhk,uhk_rb;X)
      end

      ets = online_loop.(info.online_snaps)
      res = RBResults(:u,tt,ets)
      save(info,res)
      printstyled("Average online wall time: $(tt.online_time/length(ets)) s\n";
        color=:red)

      if info.postprocess
        trian = get_triangulation(model)
        k = first(info.online_snaps)
        writevtk(info,time_info,u[k],t->U(μ[k],t),trian)
        writevtk(info,time_info,res,V,trian)
      end
    end
  end
end

if info.postprocess
  postprocess(info)
end =#
