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
partition = (10,10)
model = CartesianDiscreteModel(domain,partition)

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

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

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

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ;tt_format=true)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
println(RB.speedup(results))

save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,trial,test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
A,b = jacobian_and_residual(rbsolver,pop,smdeim)

perm = get_dof_permutation(Float64,model,test,order)

vvreffe = ReferenceFE(lagrangian,VectorValue{2,Float64},1)
vvtest = TestFESpace(model,vvreffe;conformity=:H1,dirichlet_tags=["dirichlet"])
tptest = TProductFESpace(model,vvreffe;conformity=:H1,dirichlet_tags=["dirichlet"])

import Mabla.FEM.TProduct
domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
reffe = ReferenceFE(lagrangian,Float64,2)
test = TestFESpace(model,reffe;conformity=:H1)
trial = TrialFESpace(test,x->0)
perm = TProduct.get_dof_permutation(Float64,model,test,2)

Ω = Triangulation(model)
dΩ = Measure(Ω,2)

domain1d = (0,1)
partition1d = (2,)
model1d = CartesianDiscreteModel(domain1d,partition1d)
reffe1d = ReferenceFE(lagrangian,Float64,2)
test1d = TestFESpace(model1d,reffe1d;conformity=:H1)
trial1d = TrialFESpace(test1d,x->0)
Ω1d = Triangulation(model1d)
dΩ1d = Measure(Ω1d,2)

_model = TProduct.TProductModel(domain,partition)
_test = TProduct.TProductFESpace(_model,reffe;conformity=:H1)
_perm = _test.dof_permutation

# test 1
F = assemble_vector(v->∫(v)dΩ,test)
F1d = assemble_vector(v->∫(v)dΩ1d,test1d)
TPF = kronecker(F1d,F1d)
TPF ≈ F
TPF[_perm[:]] ≈ F[perm[:]]

# test 2
f1d(x) = x[1]
f(x) = x[1]*x[2]
F = assemble_vector(v->∫(f*v)dΩ,test)
F1d = assemble_vector(v->∫(f1d*v)dΩ1d,test1d)
TPF = kronecker(F1d,F1d)
TPF ≈ F
TPF[_perm[:]] ≈ F[perm[:]]

# test 3
M = assemble_matrix((u,v)->∫(v*u)dΩ,trial,test)
M1d = assemble_matrix((u,v)->∫(v*u)dΩ1d,trial1d,test1d)
TPM = kronecker(M1d,M1d)
TPM ≈ M
TPM[_perm[:],_perm[:]] ≈ M[perm[:],perm[:]]

# test 4
M = assemble_matrix((u,v)->∫(f*v*u)dΩ,trial,test)
M1d = assemble_matrix((u,v)->∫(f1d*v*u)dΩ1d,trial1d,test1d)
TPM = kronecker(M1d,M1d)
TPM ≈ M
TPM[_perm[:],_perm[:]] ≈ M[perm[:],perm[:]]

# test 5
A = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩ,trial,test)
M1d = assemble_matrix((u,v)->∫(v*u)dΩ1d,trial1d,test1d)
A1d = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩ1d,trial1d,test1d)
TPA = kronecker(A1d,M1d) + kronecker(M1d,A1d)
TPA ≈ A
TPA[_perm[:],_perm[:]] ≈ A[perm[:],perm[:]]

x = get_cell_points(get_triangulation(test))
v = get_fe_basis(test)
v(x)

_x = get_cell_points(get_triangulation(_test))
_v = get_fe_basis(_test)
_v(_x)

_vv = _v*_v
_vv(_x)

vv = v*v
vv(x)
dvv = ∇(v)⋅∇(v)
dvv(x)

_dv = ∇(_v)
_dv(_x)
_dvv = ∇(_v)⋅∇(_v)

k = Operation(⋅)
a = ∇(_v),∇(_v)
_get_field(a::Tuple,i::Int,d::Int) = i==d ? TProduct.get_gradient_data(a[i]) : get_data(a[i])
_get_fields(a::Tuple,d::Int) = map(i->_get_field(a,i,d)[i],eachindex(a))
D = length(get_data(first(a)))
d = 1
fd = _get_fields(a,d)
evaluate!(nothing,k,fd...)
