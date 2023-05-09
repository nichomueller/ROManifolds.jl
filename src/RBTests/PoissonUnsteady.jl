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
  indef = false
  pdomain = false
  ptype = ProblemType(steady,indef,pdomain)

  mesh = "elasticity_3cyl2D.json"
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

  function a(x,p::Param,t::Real)
    exp(x[1]/sum(p.μ))
  end
  a(p::Param,t::Real) = x->a(x,p,t)
  a(p::Param) = t->a(p,t)
  m(x,p::Param,t::Real) = 1.
  m(p::Param,t::Real) = x->m(x,p,t)
  m(p::Param) = t->m(p,t)
  f(x,p::Param,t::Real) = 1.
  f(p::Param,t::Real) = x->f(x,p,t)
  f(p::Param) = t->f(p,t)
  h(x,p::Param,t::Real) = 1.
  h(p::Param,t::Real) = x->h(x,p,t)
  h(p::Param) = t->h(p,t)
  function g(x,p::Param,t::Real)
    exp(-x[1]/p.μ[2])*abs.(sin(p.μ[3]*t))*minimum(p.μ)
  end
  g(p::Param,t::Real) = x->g(x,p,t)
  g(p::Param) = t->g(p,t)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  U = ParamTransientTrialFESpace(V,g)

  feop,opA,opM,opF,opH = poisson_operators(measures,PS,time_info,V,U;a,m,f,h)

  solver = ThetaMethod(LUSolver(),dt,θ)
  nsnap = 100
  uh,μ = fe_snapshots(solver,feop,fepath,run_fem,nsnap,t0,tF;indef)

  info = RBInfoUnsteady(ptype,test_path;ϵ=1e-3,nsnap=80,mdeim_snap=20,load_offline=false,fun_mdeim=true)
  tt = TimeTracker(OfflineTime(0.,0.),0.)

  printstyled("Offline phase, reduced basis method\n";color=:blue)

  uh_offline = uh[1:info.nsnap]

  tt.offline_time.basis_time += @elapsed begin
    rbspace, = rb(info,(uh_offline,))
  end

  rbopA = RBVariable(opA,rbspace,rbspace)
  rbopM = RBVariable(opM,rbspace,rbspace)
  rbopF = RBVariable(opF,rbspace)
  rbopH = RBVariable(opH,rbspace)
  rbopAlift = RBLiftVariable(rbopA)
  rbopMlift = RBLiftVariable(rbopM)

  tt.offline_time.assembly_time = @elapsed begin
    Arb = RBAffineDecomposition(info,rbopA,measures,:dΩ,μ)
    Mrb = RBAffineDecomposition(info,rbopM,measures,:dΩ,μ)
    Frb = RBAffineDecomposition(info,rbopF,measures,:dΩ,μ)
    Hrb = RBAffineDecomposition(info,rbopH,measures,:dΓn,μ)
    Aliftrb = RBAffineDecomposition(info,rbopAlift,measures,:dΩ,μ)
    Mliftrb = RBAffineDecomposition(info,rbopMlift,measures,:dΩ,μ)
  end
  ad = (Arb,Mrb,Frb,Hrb,Aliftrb,Mliftrb)

  if info.save_offline save(info,(rbspace,ad)) end

  printstyled("Online phase, reduced basis method\n";color=:red)

  Aon = RBParamOnlineStructure(Arb;st_mdeim=info.st_mdeim)
  Mon = RBParamOnlineStructure(Mrb;st_mdeim=info.st_mdeim)
  Fon = RBParamOnlineStructure(Frb;st_mdeim=info.st_mdeim)
  Hon = RBParamOnlineStructure(Hrb;st_mdeim=info.st_mdeim)
  Alifton = RBParamOnlineStructure(Aliftrb;st_mdeim=info.st_mdeim)
  Mlifton = RBParamOnlineStructure(Mliftrb;st_mdeim=info.st_mdeim)
  param_on_structures = (Aon,Mon,Fon,Hon,Alifton,Mlifton)

  err = ErrorTracker[]
  function online_loop(k::Int)
    tt.online_time += @elapsed begin
      lhs,rhs = unsteady_poisson_rb_system(param_on_structures,μ[k])
      rb_sol = solve_rb_system(lhs,rhs)
    end
    uhk = uh[k]
    uhk_rb = reconstruct_fe_sol(rbspace,rb_sol)
    push!(err,ErrorTracker(uhk,uhk_rb))
  end

  pmap(online_loop,info.online_snaps)
  res = RBResults(:u,tt,err)
  postprocess(info,(res,),(V,),model,time_info)
end

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
  uh,μ = fe_snapshots(ptype,solver,op,fepath,run_fem,nsnap,t0,tF)

  opA = NonaffineParamOperator(a,afe,PS,time_info,U,V;id=:A)
  opM = AffineParamOperator(m,mfe,PS,time_info,U,V;id=:M)
  opF = AffineParamOperator(f,ffe,PS,time_info,V;id=:F)
  opH = AffineParamOperator(h,hfe,PS,time_info,V;id=:H)

for fun_mdeim = (false)#(false,true)
  for st_mdeim = (false)#(false,true)
    for tol = (1e-4)#(1e-2,1e-3,1e-4,1e-5)

      global info = RBInfoUnsteady(ptype,test_path;ϵ=tol,nsnap=80,online_snaps=95:100,
        mdeim_snap=20,load_offline=false,save_offline=false,postprocess=true,
        fun_mdeim=fun_mdeim,st_mdeim=st_mdeim,save_online=false)
      tt = TimeTracker(OfflineTime(0.,0.),0.)

      printstyled("Offline phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:blue)

      uh_offline = uh[1:info.nsnap]
      #X = H1_norm_matrix(opA,opM)
      rbspace = rb(info,tt,uh_offline)#;X)

      rbopA = RBVariable(opA,rbspace,rbspace)
      rbopM = RBVariable(opM,rbspace,rbspace)
      rbopF = RBVariable(opF,rbspace)
      rbopH = RBVariable(opH,rbspace)

      Arb = RBAffineDecomposition(info,tt,rbopA,μ,measures,:dΩ)
      Mrb = RBAffineDecomposition(info,tt,rbopM,μ,measures,:dΩ)
      Frb = RBAffineDecomposition(info,tt,rbopF,μ,measures,:dΩ)
      Hrb = RBAffineDecomposition(info,tt,rbopH,μ,measures,:dΓn)

      ad = (Arb,Mrb,Frb,Hrb)
      ad_eval = eval_affine_decomposition(ad)
      param_on_structures = RBParamOnlineStructure(ad,ad_eval;st_mdeim=info.st_mdeim)

      printstyled("Online phase; tol=$tol, st_mdeim=$st_mdeim, fun_mdeim=$fun_mdeim\n";color=:red)

      function online_loop(k::Int)
        printstyled("-------------------------------------------------------------\n")
        printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
        tt.online_time += @elapsed begin
          printstyled("Evaluating RB system for μ = μ[$k]\n";color=:red)
          lhs,rhs = unsteady_poisson_rb_system(param_on_structures,μ[k])
          rb_sol = solve_rb_system(lhs,rhs)
        end
        uhk = get_snap(uh[k])
        uhk_rb = reconstruct_fe_sol(rbspace,rb_sol)
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
        writevtk(info,time_info,uh[k],t->U(μ[k],t),trian)
        writevtk(info,time_info,res,V,trian)
      end
    end
  end
end

if info.postprocess
  postprocess(info)
end =#
