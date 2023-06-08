# driver for unsteady poisson
using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(manager)

root = pwd()
include("$root/src/NEWCODE/FEM/FEM.jl")
include("$root/src/NEWCODE/ROM/ROM.jl")
include("$root/src/NEWCODE/RBTests.jl")

mesh = "flow_3cyl2D.json"
test_path = "$root/tests/stokes/unsteady/$mesh"
bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
order = 2
degree = 4

fepath = fem_path(test_path)
mshpath = mesh_path(test_path,mesh)
model = get_discrete_model(mshpath,bnd_info)
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

ranges = fill([1.,10.],3)
sampling = UniformSampling()
pspace = ParamSpace(ranges,sampling)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)

g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
g(μ,t) = x->g(x,μ,t)
g0(x,μ,t) = VectorValue(0,0)
g0(μ,t) = x->g0(x,μ,t)

lhs_t(μ,t,(u,p),(v,q)) = ∫(v⋅u)dΩ
lhs(μ,t,(u,p),(v,q)) = ∫(a(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
rhs(μ,t,(v,q)) = ∫(g0(μ,t)⋅v)dΩ

reffe_u = Gridap.ReferenceFE(lagrangian,VectorValue{2,Float},order)
reffe_p = Gridap.ReferenceFE(lagrangian,Float,order-1)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
trial_u = ParamTransientTrialFESpace(test_u,[g0,g])
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = ParamTransientMultiFieldFESpace([test_u,test_p])
trial = ParamTransientMultiFieldFESpace([trial_u,trial_p])
feop = ParamTransientAffineFEOperator(lhs_t,lhs,rhs,pspace,trial,test)
t0,tF,dt,θ = 0.,0.3,0.005,1
fesolver = θMethod(LUSolver(),t0,tF,dt,θ)

ϵ = 1e-4
load_offline = false
energy_norm = false
nsnaps = 10
mdeim_nsnaps = 10
info = RBInfo(test_path;ϵ,load_offline,energy_norm,nsnaps,mdeim_nsnaps)

rbspace = reduce_fe_space(info,feop,fesolver)
rbop = reduce_fe_operator(info,feop,fesolver,rbspace)
rbsolver = Backslash()

u_rb = solve(rbsolver,rbop;n_solutions=10,post_process=true,energy_norm)
