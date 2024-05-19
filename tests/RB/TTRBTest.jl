using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.Helpers
using Gridap.TensorValues
using BlockArrays
using DrWatson
using Kronecker
using Mabla.FEM
using Mabla.RB

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.05

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (5,5)
model = TProductModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

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

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=2)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
# norm_matrix = assemble_norm_matrix(feop)
# cores = ttsvd(soff,norm_matrix)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
A,b = jacobian_and_residual(rbsolver,pop,smdeim)

b1,t1 = b.values[1],b.trians[1]
# bred = RB.reduced_form(rbsolver,b1,t1,get_test(pop))
mdeim_style = rbsolver.mdeim_style
basis = reduced_basis(b1;ϵ=RB.get_tol(rbsolver))
lu_interp,integration_domain = mdeim(mdeim_style,basis)
proj_basis = reduce_operator(mdeim_style,basis,get_test(pop))

b_test = RB.get_basis(get_test(pop))

STOP
# fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
# results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

# println(RB.space_time_error(results))
# println(RB.speedup(results))

# save(test_dir,fesnaps)
# save(test_dir,rbop)
# save(test_dir,results)

form(u,v) = ∫(u*v)dΩ
M = assemble_matrix(form,trial(nothing),test)
p = FEM.get_free_dof_permutation(test)
_Mp = M[p,p]
Mp = permutedims(M[p,p],[1,3,2,4])

boh = rand(4,5,4,5)
permutedims(boh,[1,3,2,4])

using SparseArrays
I,J,V = findnz(M)

using PartitionedArrays
Ix,Iy = tuple_of_arrays(map(i->Tuple(findfirst(p .== i)),I))
Jx,Jy = tuple_of_arrays(map(i->Tuple(findfirst(p .== i)),J))
Ixy = map(i->findfirst(p .== i),I)
Jxy = map(i->findfirst(p .== i),J)
psparseI = p[Ixy]
# ci = map(CartesianIndex,I,J)

ids = CartesianIndices(size(p))

M = assemble_norm_matrix(form,trial(nothing),test)
Mk = kronecker(M.arrays_1d[1],M.arrays_1d[2])
