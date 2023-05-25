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

function stokes_unsteady()
  run_fem = false

  steady = false
  indef = true
  ptype = ProblemType(steady,indef)

  mesh = "flow_3cyl_015.json"
  test_path = "$root/tests/stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet_noslip" => ["noslip"],
                  "dirichlet_nopenetration" => ["nopenetration"],
                  "dirichlet" => ["inlet"],
                  "neumann" => ["outlet"])
  order = 2

  fepath = fem_path(test_path)
  mshpath = mesh_path(test_path,mesh)
  model = model_info(mshpath,bnd_info)
  measures = ProblemMeasures(model,order)
  D = get_dimension(model)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  PS = ParamSpace(ranges,sampling)

  t0,tF,dt,θ = 0.,0.15,0.0025,1
  time_info = ThetaMethodInfo(t0,tF,dt,θ)

  function a(x,p::Param,t::Real)
    exp((sin(t)+cos(t))*x[1]/sum(p.μ))
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
    flow_rate = abs(1-cos(pi*t/(2*tF))+sin(pi*t/(2*p.μ[1]*tF))/p.μ[2])
    parab_prof = p.μ[3]*VectorValue(abs.(x[2]*(x[2]-W)),0.,0.)
    parab_prof*flow_rate
  end
  g(p::Param,t::Real) = x->g(x,p,t)
  g0(x,p::Param,t::Real) = VectorValue(0.,0.,0.)
  g0(p::Param,t::Real) = x->g0(x,p,t)

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

  # Remote generation of FEM snapshots; the task is launched on the local process,
  # and split among on all available remote workers thanks to a pmap.
  # Then, the snapshots are sent to the remote workers

  u,p,μ = generate_fe_snapshots(Val{indef}(),run_fem,fepath,nsnap,solver,feop,t0,tF)

  for fun_mdeim=(true,), st_mdeim=(false,), ϵ=(1e-1,1e-2,1e-3,1e-4)
    info = RBInfoUnsteady(ptype,test_path;ϵ,nsnap=50,mdeim_snap=30,
      st_mdeim,fun_mdeim,postprocess=true)

    printstyled("Offline phase, reduced basis method\n";color=:blue)

    u_off = u[1:info.nsnap]
    p_off = p[1:info.nsnap]

    basis_time = @elapsed begin
      rbspace_u,rbspace_p = assemble_rb_space(info,(u_off,p_off),opB)
    end
    rb_space = rbspace_u,rbspace_p

    rbopA = RBVariable(opA,rbspace_u,rbspace_u)
    rbopB = RBVariable(opB,rbspace_p,rbspace_u)
    rbopBT = RBVariable(opBT,rbspace_u,rbspace_p)
    rbopM = RBVariable(opM,rbspace_u,rbspace_u)
    rbopF = RBVariable(opF,rbspace_u)
    rbopH = RBVariable(opH,rbspace_u)
    rbopAlift = RBLiftVariable(rbopA)
    rbopBlift = RBLiftVariable(rbopB)
    rbopMlift = RBLiftVariable(rbopM)

    assembly_time = @elapsed begin
      Arb = RBAffineDecomposition(info,rbopA,μ,get_dΩ(measures))
      Brb = RBAffineDecomposition(info,rbopB,μ,get_dΩ(measures))
      BTrb = RBAffineDecomposition(info,rbopBT,μ,get_dΩ(measures))
      Mrb = RBAffineDecomposition(info,rbopM,μ,get_dΩ(measures))
      Frb = RBAffineDecomposition(info,rbopF,μ,get_dΩ(measures))
      Hrb = RBAffineDecomposition(info,rbopH,μ,get_dΓn(measures))
      Aliftrb = RBAffineDecomposition(info,rbopAlift,μ,get_dΩ(measures))
      Bliftrb = RBAffineDecomposition(info,rbopBlift,μ,get_dΩ(measures))
      Mliftrb = RBAffineDecomposition(info,rbopMlift,μ,get_dΩ(measures))
    end
    adrb = (Arb,Brb,BTrb,Mrb,Frb,Hrb,Aliftrb,Bliftrb,Mliftrb)

    offline_times = OfflineTime(basis_time,assembly_time)

    if info.save_offline
      save(info,(rb_space,adrb,offline_times))
    end

    printstyled("Online phase, reduced basis method\n";color=:red)

    Aon = RBParamOnlineStructure(Arb;st_mdeim)
    Bon = RBParamOnlineStructure(Brb;st_mdeim)
    BTon = RBParamOnlineStructure(BTrb;st_mdeim)
    Mon = RBParamOnlineStructure(Mrb;st_mdeim)
    Fon = RBParamOnlineStructure(Frb;st_mdeim)
    Hon = RBParamOnlineStructure(Hrb;st_mdeim)
    Alifton = RBParamOnlineStructure(Aliftrb;st_mdeim)
    Blifton = RBParamOnlineStructure(Bliftrb;st_mdeim)
    Mlifton = RBParamOnlineStructure(Mliftrb;st_mdeim)
    online_structures = (Aon,Bon,BTon,Mon,Fon,Hon,Alifton,Blifton,Mlifton)

    u_on = convert_snapshot(Matrix{Float},u)
    p_on = convert_snapshot(Matrix{Float},p)
    μ_on = μ[info.online_snaps]
    rb_system = k -> unsteady_stokes_rb_system(online_structures,μ[k])
    online_loop_k = k -> online_loop((u_on[k],p_on[k]),rb_space,rb_system,k)
    res = online_loop(online_loop_k,info.online_snaps)
    postprocess(info,res,
    ((t->U(μ_on[1],t),V),(P,Q)),model,time_info)
  end
end

stokes_unsteady()
