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

# Setting up the problem on all processes
@everywhere begin
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

  info = RBInfoUnsteady(ptype,test_path;ϵ=1e-4,nsnap=80,mdeim_snap=20)
end

# Remote generation of FEM snapshots; the task is launched on the local process,
# and split among on all available remote workers thanks to a pmap.
# Then, the snapshots are sent to the remote workers
begin
  u,p,μ = generate_fe_snapshots_on_workers(run_fem,fepath,nsnap,solver,feop,t0,tF;indef)
  @passobj 1 workers() u
  @passobj 1 workers() p
  @passobj 1 workers() μ
end

# Generation of the reduced basis in parallel on the remote workers, leveraging
# the Elemental package
@mpi_do manager begin
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
end

# Remote generation of MDEIM snapshots with parallel map on all remote workers
begin
  rbopA = (@getfrom first(workers()) rbopA)::RBUnsteadyBilinVariable
  rbopB = (@getfrom first(workers()) rbopB)::RBUnsteadyBilinVariable
  rbopBT = (@getfrom first(workers()) rbopBT)::RBUnsteadyBilinVariable
  rbopM = (@getfrom first(workers()) rbopM)::RBUnsteadyBilinVariable
  rbopF = (@getfrom first(workers()) rbopF)::RBUnsteadyLinVariable
  rbopH = (@getfrom first(workers()) rbopH)::RBUnsteadyLinVariable
  rbopAlift = (@getfrom first(workers()) rbopAlift)::RBUnsteadyLiftVariable
  rbopBlift = (@getfrom first(workers()) rbopBlift)::RBUnsteadyLiftVariable
  rbopMlift = (@getfrom first(workers()) rbopAlift)::RBUnsteadyLiftVariable

  μ_off = μ[1:info.mdeim_nsnap]

  assembly_time = @elapsed begin
    A = AffineDecomposition(info,rbopA,μ_off)
    B = AffineDecomposition(info,rbopB,μ_off)
    BT = AffineDecomposition(info,rbopBT,μ_off)
    M = AffineDecomposition(info,rbopM,μ_off)
    F = AffineDecomposition(info,rbopF,μ_off)
    H = AffineDecomposition(info,rbopH,μ_off)
    Alift = AffineDecomposition(info,rbopAlift,μ_off)
    Blift = AffineDecomposition(info,rbopBlift,μ_off)
    Mlift = AffineDecomposition(info,rbopMlift,μ_off)
  end

  @passobj 1 workers() assembly_time
  @passobj 1 workers() A
  @passobj 1 workers() B
  @passobj 1 workers() BT
  @passobj 1 workers() M
  @passobj 1 workers() F
  @passobj 1 workers() H
  @passobj 1 workers() Alift
  @passobj 1 workers() Blift
  @passobj 1 workers() Mlift
end

@mpi_do manager begin
  assembly_time = @elapsed begin
    Arb = RBAffineDecomposition(info,rbopA,A,measures,:dΩ)
    Brb = RBAffineDecomposition(info,rbopB,B,measures,:dΩ)
    BTrb = RBAffineDecomposition(info,rbopBT,BT,measures,:dΩ)
    Mrb = RBAffineDecomposition(info,rbopM,M,measures,:dΩ)
    Frb = RBAffineDecomposition(info,rbopF,F,measures,:dΩ)
    Hrb = RBAffineDecomposition(info,rbopH,H,measures,:dΓn)
    Aliftrb = RBAffineDecomposition(info,rbopAlift,Alift,measures,:dΩ)
    Bliftrb = RBAffineDecomposition(info,rbopBlift,Blift,measures,:dΩ)
    Mliftrb = RBAffineDecomposition(info,rbopMlift,Mlift,measures,:dΩ)
  end
  adrb = (Arb,Brb,BTrb,Mrb,Frb,Hrb,Aliftrb,Bliftrb,Mliftrb)

  offline_times = OfflineTime(basis_time,assembly_time)

  if info.save_offline
    save(info,(rb_space,adrb,offline_times))
  end

# The method's online phase can be carried out on the local process; a pmap is
# used to loop over the online parameters
begin
  printstyled("Online phase, reduced basis method\n";color=:red)

  u = (@getfrom first(workers()) u)::Snapshots
  p = (@getfrom first(workers()) p)::Snapshots
  μ = (@getfrom first(workers()) μ)::Vector{Param}
  rb_space = (@getfrom first(workers()) rb_space)::NTuple{2,RBSpaceUnsteady}
  Arb = (@getfrom first(workers()) Arb)::RBAffineDecomposition
  Brb = (@getfrom first(workers()) Brb)::RBAffineDecomposition
  BTrb = (@getfrom first(workers()) BTrb)::RBAffineDecomposition
  Mrb = (@getfrom first(workers()) Mrb)::RBAffineDecomposition
  Frb = (@getfrom first(workers()) Frb)::RBAffineDecomposition
  Hrb = (@getfrom first(workers()) Hrb)::RBAffineDecomposition
  Aliftrb = (@getfrom first(workers()) Aliftrb)::RBAffineDecomposition
  Bliftrb = (@getfrom first(workers()) Bliftrb)::RBAffineDecomposition
  Mliftrb = (@getfrom first(workers()) Mliftrb)::RBAffineDecomposition

  st_mdeim = info.st_mdeim
  Aon = RBParamOnlineStructure(Arb;st_mdeim)
  Bon = RBParamOnlineStructure(Brb;st_mdeim)
  BTon = RBParamOnlineStructure(BTrb;st_mdeim)
  Mon = RBParamOnlineStructure(Mrb;st_mdeim)
  Fon = RBParamOnlineStructure(Frb;st_mdeim)
  Hon = RBParamOnlineStructure(Hrb;st_mdeim)
  Alifton = RBParamOnlineStructure(Aliftrb;st_mdeim)
  Mlifton = RBParamOnlineStructure(Mliftrb;st_mdeim)
  online_structures = (Aon,Bon,BTon,Mon,Fon,Hon,Alifton,Blifton,Mlifton)

  @passobj 1 workers() online_structures
  @everywhere rb_system(k) = unsteady_stokes_rb_system(online_structures,μ[k])
  @everywhere loop(k) = online_loop(k->(u[k],p[k]),rb_space,rb_system,k)
  res = online_loop(loop,info.online_snaps)
  postprocess(info,res,(V,Q),model,time_info)
end
