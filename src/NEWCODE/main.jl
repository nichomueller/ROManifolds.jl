# driver for unsteady poisson
using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(manager)

root = pwd()
include("$root/src/FEM/FEM.jl")
include("$root/src/RB/RB.jl")
include("$root/src/RBTests/RBTests.jl")

mesh = "elasticity_3cyl.json"
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
PS = ParamSpace(ranges,sampling)

t0,tF,dt,θ = 0.,0.3,0.005,1
time_info = ThetaMethodInfo(t0,tF,dt,θ)

a(x,p::Param,t::Real) = exp((sin(t)+cos(t))*x[1]/sum(p.μ))
a(p::Param,t::Real) = x->a(x,p,t)

f(x,p::Param,t::Real) = 1.
f(p::Param,t::Real) = x->f(x,p,t)

h(x,p::Param,t::Real) = abs(cos(p.μ[3]*t))
h(p::Param,t::Real) = x->h(x,p,t)

g(x,p::Param,t::Real) = p.μ[1]*exp(-x[1]/p.μ[2])*abs(sin(p.μ[3]*t))
g(p::Param,t::Real) = x->g(x,p,t)

lhs_t(u,v) = ∫(v*u)dΩ
lhs(p,t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
rhs(p,t,v) = ∫(f(p,t)*v)dΩ + ∫(h(p,t)*v)dΓn

reffe = Gridap.ReferenceFE(lagrangian,Float,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = ParamTransientTrialFESpace(test,g)
feop = ParamTransientAffineFEOperator(lhs_t,lhs,rhs,pspace,test,trial)
fesolver = TimeMarchingScheme(LUSolver(),time_info)

load_offline = false
ϵ = 1e-3
energy_norm = false
pod_style = ReducedPOD()

rbspace = reduce_fe_space(feop;load_offline,fesolver,n_snaps=80,ϵ,energy_norm,pod_style)
rbop = reduce_fe_operator(feop,rbspace;load_offline,n_snaps=20,ϵ)
rbsolver = Backslash()

u_rb = solve(rbsolver,rbop;n_solutions=10,post_process=true,energy_norm)
