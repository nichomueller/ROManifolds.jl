# Init MPI
using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(manager)

# Loading packages on all processes
@everywhere begin
  root = pwd()
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
  include("$root/src/RBTests/RBTests.jl")
end

@everywhere function poisson_unsteady()
  run_fem = false

  steady = false
  indef = false
  ptype = ProblemType(steady,indef)

  mesh = "elasticity_3cyl.json"
  test_path = "$root/tests/poisson/unsteady/$mesh"
  bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
  order = 1

  fepath = fem_path(test_path)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info)
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

  # Remote generation of FEM snapshots; the task is launched on the local process,
  # and split among on all available remote workers thanks to a pmap.
  # Then, the snapshots are sent to the remote workers
  u,μ = generate_fe_snapshots(Val{indef}(),run_fem,fepath,nsnap,solver,feop,t0,tF)

  for fun_mdeim=(true,false), st_mdeim=(true,false), ϵ=(1e-4,)#ϵ=(1e-1,1e-2,1e-3,1e-4)
    info = RBInfoUnsteady(ptype,test_path;ϵ,nsnap=80,mdeim_snap=30,
      st_mdeim,fun_mdeim,postprocess=true,load_offline=true)

    printstyled("Offline phase, reduced basis method\n";color=:blue)

    u_offline = u[1:info.nsnap]

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
      Arb = RBAffineDecomposition(info,rbopA,μ,get_dΩ(measures))
      Mrb = RBAffineDecomposition(info,rbopM,μ,get_dΩ(measures))
      Frb = RBAffineDecomposition(info,rbopF,μ,get_dΩ(measures))
      Hrb = RBAffineDecomposition(info,rbopH,μ,get_dΓn(measures))
      Aliftrb = RBAffineDecomposition(info,rbopAlift,μ,get_dΩ(measures))
      Mliftrb = RBAffineDecomposition(info,rbopMlift,μ,get_dΩ(measures))
    end
    adrb = (Arb,Mrb,Frb,Hrb,Aliftrb,Mliftrb)

    offline_times = OfflineTime(basis_time,assembly_time)

    if info.save_offline save(info,(rb_space,adrb,offline_times)) end

    printstyled("Online phase, reduced basis method\n";color=:red)

    Aon = RBParamOnlineStructure(Arb;st_mdeim)
    Mon = RBParamOnlineStructure(Mrb;st_mdeim)
    Fon = RBParamOnlineStructure(Frb;st_mdeim)
    Hon = RBParamOnlineStructure(Hrb;st_mdeim)
    Alifton = RBParamOnlineStructure(Aliftrb;st_mdeim)
    Mlifton = RBParamOnlineStructure(Mliftrb;st_mdeim)
    online_structures = (Aon,Mon,Fon,Hon,Alifton,Mlifton)

    u_on = convert_snapshot(Matrix{Float},u)
    μ_on = μ[info.online_snaps]
    rb_system = k -> unsteady_poisson_rb_system(online_structures,μ[k])
    online_loop_k = k -> online_loop(u_on[k],rb_space,rb_system,k)
    res = online_loop(online_loop_k,info.online_snaps)
    postprocess(info,(res,),((t->U(μ_on[1],t),V),),model,time_info)
  end
end

poisson_unsteady()
