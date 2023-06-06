# driver for unsteady poisson
using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(manager)

root = pwd()
include("$root/src/NEWCODE/FEM/FEM.jl")
include("$root/src/NEWCODE/ROM/ROM.jl")
include("$root/src/NEWCODE/RBTests.jl")

mesh = "elasticity_3cyl2D.json"
test_path = "$root/tests/poisson/unsteady/$mesh"
bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
order = 1
degree = 2

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

t0,tF,dt,θ = 0.,0.3,0.005,1
time_info = ThetaMethodInfo(t0,tF,dt,θ)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)

h(x,μ,t) = abs(cos(μ[3]*t))
h(μ,t) = x->h(x,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]*t))
g(μ,t) = x->g(x,μ,t)

lhs_t(μ,t,u,v) = ∫(v*u)dΩ
lhs(μ,t,u,v) = ∫(a(μ,t)*∇(v)⋅∇(u))dΩ
rhs(μ,t,v) = ∫(f(μ,t)*v)dΩ + ∫(h(μ,t)*v)dΓn

reffe = Gridap.ReferenceFE(lagrangian,Float,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = ParamTransientTrialFESpace(test,g)
feop = ParamTransientAffineFEOperator(lhs_t,lhs,rhs,pspace,trial,test)
fesolver = ThetaMethod(LUSolver(),dt,θ)

load_offline = false
ϵ = 1e-3
energy_norm = false
pod_style = ReducedPOD()

rbspace = reduce_fe_space(feop,fesolver,t0,tF;load_offline,n_snaps=80,ϵ,energy_norm,pod_style)
rbop = reduce_fe_operator(feop,rbspace;load_offline,n_snaps=20,ϵ)
rbsolver = Backslash()

u_rb = solve(rbsolver,rbop;n_solutions=10,post_process=true,energy_norm)
