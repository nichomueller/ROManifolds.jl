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
afe(p::Param,t::Real,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
afe(p::Param,t::Real) = (u,v) -> afe(p,t,u,v)

m(x,p::Param,t::Real) = 1.
m(p::Param,t::Real) = x->m(x,p,t)
mfe(p::Param,t::Real,u,v) = ∫(m(p,t)*v*u)dΩ
mfe(p::Param,t::Real) = (u,v) -> mfe(p,t,u,v)

f(x,p::Param,t::Real) = 1.
f(p::Param,t::Real) = x->f(x,p,t)
ffe(p::Param,t::Real,v) = ∫(f(p,t)*v)dΩ
ffe(p::Param,t::Real) = v -> ffe(p,t,v)

h(x,p::Param,t::Real) = abs(cos(p.μ[3]*t))
h(p::Param,t::Real) = x->h(x,p,t)
hfe(p::Param,t::Real,v) = ∫(h(p,t)*v)dΓn
hfe(p::Param,t::Real) = v -> hfe(p,t,v)

g(x,p::Param,t::Real) = p.μ[1]*exp(-x[1]/p.μ[2])*abs(sin(p.μ[3]*t))
g(p::Param,t::Real) = x->g(x,p,t)

lhs_t(p,t,u,v) = mfe(p,t,u,v)
lhs(p,t,u,v) = afe(p,t,u,v)
rhs(p,t,v) = ffe(p,t,v) + hfe(p,t,v)

feop = ParamTransientAffineFEOperator(lhs_t,lhs,rhs,PS,U,V)

reffe = Gridap.ReferenceFE(lagrangian,Float,order)
V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
U = ParamTransientTrialFESpace(V,g)

ϵ = 1e-3
energy_norm = false

Vrb,Urb = reduce_fe_spaces(feop,V,U;n_snapshots=80,ϵ,energy_norm)
rbop = reduce_fe_operator(feop,Vrb,Urb;n_snapshots=20,ϵ)
solver = Backslash()

u_rb = solve(solver,rbop;n_solutions=10,post_process=true,energy_norm)
