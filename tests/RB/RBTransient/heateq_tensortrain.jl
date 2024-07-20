using Gridap
using Test
using DrWatson
using Serialization

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

# time marching
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# geometry
n = 5
domain = (0,1,0,1)
partition = (n,n)
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

# weak formulation
a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

induced_norm(du,v) = ∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)
ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(compute_error(results))
println(compute_speedup(results))

using Gridap.CellData
using Gridap.FESpaces
using Gridap.ODEs
using Mabla.FEM.IndexMaps

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
op = get_algebraic_operator(feop)
op = TransientPGOperator(op,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RBSteady.mdeim_params(rbsolver))
j,r = jacobian_and_residual(rbsolver,op,smdeim)
red_jac = reduced_jacobian(rbsolver,op,j)
red_res = reduced_residual(rbsolver,op,r)

j1 = j[1][1]
basis = reduced_basis(j1)
lu_interp,integration_domain = mdeim(rbsolver.mdeim_style,basis)

cores_space = get_cores_space(basis)
core_time = RBTransient.get_core_time(basis)
_indices_spacetime,_interp_basis_spacetime = empirical_interpolation(cores_space...,core_time)

i = get_index_map(j1)
bst = RBTransient.get_basis_spacetime(i,cores_space,core_time)
# indices_spacetime,interp_basis_spacetime = empirical_interpolation(bst)

b1 = vec(select_snapshots(j1,1))
b1 - bst*(_interp_basis_spacetime\b1[_indices_spacetime])

function old_mdeim(mdeim_style,b::TransientTTSVDCores)
  i = get_index_map(b)
  bst = RBTransient.get_basis_spacetime(i,get_cores_space(b),RBTransient.get_core_time(b))
  indices_spacetime,interp_basis_spacetime = empirical_interpolation(bst)
  indices_space = fast_index(indices_spacetime,num_space_dofs(b))
  indices_time = slow_index(indices_spacetime,num_space_dofs(b))
  lu_interp = lu(interp_basis_spacetime)
  integration_domain = TransientIntegrationDomain(indices_space,indices_time)
  return lu_interp,integration_domain
end

old_lu_interp,old_integration_domain = old_mdeim(rbsolver.mdeim_style,basis)

_get_array(c::AbstractArray) = c
_get_array(c::SparseCore) = c.array
_recast_indices!(i,c::AbstractArray) = i
_recast_indices!(i,c::SparseCore) = recast_indices!(i,c.sparsity.matrix)
_get_size(c::AbstractArray) = size(c,2)
_get_size(c::SparseCore) = size(c,2)*size(c,2)

cores = get_cores(basis)

C = _get_array(first(cores))
A = dropdims(C;dims=1)
m,n = size(A)
r = zeros(eltype(A),m)
I′ = zeros(Int32,n)
I = Vector{Int32}[]
s = Int32[]
for i = eachindex(cores)
  I′,Ai = RBSteady.empirical_interpolation!((I′,r),A)
  _recast_indices!(I′,cores[i])
  push!(I,copy(I′))
  push!(s,_get_size(cores[i]))
  if i < length(cores)
    Ci = reshape(Ai,1,size(Ai)...)
    A = cores2basis(Ci,_get_array(cores[i+1]))
  else
    Ig = _to_global_indices(I,s)
    return Ig,Ai
  end
end
