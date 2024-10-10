using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
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

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

n = 10
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[6,])
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

order = 2
degree = 2*(order)+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ) = 1
a(μ) = x->a(x,μ)
aμ(μ) = ParamFunction(a,μ)

g(x,μ) = VectorValue(μ[1],0.0)
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)
g0(x,μ) = VectorValue(0.0,0.0)
g0(μ) = x->g0(x,μ)
gμ0(μ) = ParamFunction(g0,μ)

const Re = 10.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

jac_lin(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
res_lin(μ,(u,p),(v,q),dΩ) = jac_lin(μ,(u,p),(v,q),dΩ)

jac_nlin(μ,(u,p),(du,dp),(v,q),dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ
res_nlin(μ,(u,p),(v,q),dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ

trian_res = (Ω,)
trian_jac = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["diri0","diri1"])
trial_u = ParamTrialFESpace(test_u,[gμ0,gμ])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop_lin = LinearParamFEOperator(res_lin,jac_lin,pspace,trial,test,trian_res,trian_jac)
feop_nlin = ParamFEOperator(res_nlin,jac_nlin,pspace,trial,test,trian_res,trian_jac)
feop = LinNonlinParamFEOperator(feop_lin,feop_nlin)

fesolver = NonlinearFESolver(NewtonRaphsonSolver(LUSolver(),1e-10,20))

tol = 1e-4
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=30,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=10,nparams_jac=10)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,rbop,x,x̂,festats,rbstats,μon)
