# driver for unsteady poisson
using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(manager)

@everywhere begin
  root = pwd()
  include("$root/src/NEWCODE/FEM/FEM.jl")
  include("$root/src/NEWCODE/ROM/ROM.jl")
  include("$root/src/NEWCODE/RBTests.jl")
end

mesh = "elasticity_3cyl2D.json"
test_path = "$root/tests/poisson/unsteady/$mesh"
bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
order = 1
degree = 2

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

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)

h(x,μ,t) = abs(cos(μ[3]*t))
h(μ,t) = x->h(x,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]*t))
g(μ,t) = x->g(x,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)

lhs_t(μ,t,u,v) = ∫(v*u)dΩ
lhs(μ,t,u,v) = ∫(a(μ,t)*∇(v)⋅∇(u))dΩ
rhs(μ,t,v) = ∫(f(μ,t)*v)dΩ + ∫(h(μ,t)*v)dΓn

reffe = Gridap.ReferenceFE(lagrangian,Float,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = ParamTransientTrialFESpace(test,g)
feop = ParamTransientAffineFEOperator(lhs_t,lhs,rhs,pspace,trial,test)
t0,tF,dt,θ = 0.,0.05,0.005,1
uh0(μ) = interpolate_everywhere(u0(μ),trial(μ,t0))
fesolver = θMethod(LUSolver(),t0,tF,dt,θ,uh0)

ϵ = 1e-4
save_offline = false
load_offline = false
energy_norm = false
nsnaps = 10
nsnaps_mdeim = 10
info = RBInfo(test_path;ϵ,load_offline,save_offline,energy_norm,nsnaps,nsnaps_mdeim)

rbspace = reduce_fe_space(info,feop,fesolver)

times = get_times(solver)
sols,param = get_data(s)
sols = convert(Matrix{Float},sols.array)
matdatum = _matdata_jacobian(feop,solver,sols,param)
aff = Affinity(matdatum,param,times)
matdata = map(μ -> map(t -> matdatum(μ,t),times),param)
A = _nonaffine_jacobian(op.assem,matdata,filter)

vecdatum = _vecdata_residual(op,solver,sols,param)
aff = Affinity(vecdatum,param,times)
vecdata = map(μ -> map(t -> vecdatum(μ,t),times),param)
b = _nonaffine_residual(op.assem,vecdata,filter)

# if load_offline
#   rbop = load(RBOperator,info)
# else
#   rbspace = reduce_fe_space(info,feop,fesolver)
#   rbop = reduce_fe_operator(info,feop,fesolver,rbspace)
# end

# rbsolver = Backslash()
# u_rb = solve(info,rbsolver,rbop;n_solutions=10,post_process=true,energy_norm)
