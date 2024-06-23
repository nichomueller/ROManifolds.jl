using Gridap
using Gridap.MultiField
using Test
using DrWatson

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

θ = 1.0
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
# model_dir = datadir(joinpath("meshes","perforated_plate.json"))
# model = DiscreteModelFromFile(model_dir)

n = 10
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = 1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

# inflow(μ,t) = 1-cos(2π*t/tf)+sin(μ[2]*2π*t/tf)/μ[1]
inflow(μ,t) = 1-cos(2π*t/(μ[1]*tf))
g_in_1(x,μ,t) = -x[2]*(1-x[2])*inflow(μ,t)
g_in_1(μ,t) = x->g_in_1(x,μ,t)
gμt_in_1(μ,t) = TransientParamFunction(g_in_1,μ,t)
g_in_2(x,μ,t) = 0.0
g_in_2(μ,t) = x->g_in_1(x,μ,t)
gμt_in_2(μ,t) = TransientParamFunction(g_in_2,μ,t)
gμt_w_1 = gμt_in_2
gμt_w_2 = gμt_in_2
gμt_c_1 = gμt_in_2
gμt_c_2 = gμt_in_2

u0_1(x,μ) = 0.0
u0_1(μ) = x->u0_1(x,μ)
u0μ_1(μ) = ParamFunction(u0_1,μ)
u0μ_2 = u0μ_1
p0μ = u0μ_1

a_11(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
a_12(μ,t,p,v,dΩ) = ∫(p*(∇⋅(v)))dΩ
a_21(μ,t,u,q,dΩ) = ∫(q*(∇⋅(u)))dΩ

stiffness(μ,t,(u_1,u_2,p),(v_1,v_2,q),dΩ) = (
  a_11(μ,t,u_1,v_1,dΩ) + a_11(μ,t,u_1,v_2,dΩ) + a_11(μ,t,u_2,v_1,dΩ) + a_11(μ,t,u_2,v_2,dΩ)
  - a_12(μ,t,p,v_1,dΩ) - a_12(μ,t,p,v_2,dΩ)
  + a_21(μ,t,u_1,q,dΩ) + a_21(μ,t,u_2,q,dΩ)
  )
mass(μ,t,(uₜ_1,uₜ_2,pₜ),(v_1,v_2,q),dΩ) = ∫(v_1*uₜ_1)dΩ + ∫(v_2*uₜ_1)dΩ + ∫(v_1*uₜ_2)dΩ + ∫(v_2*uₜ_2)dΩ
res(μ,t,(u_1,u_2,p),(v_1,v_2,q),dΩ) = (
  mass(μ,t,(∂t(u_1),∂t(u_2),p),(v_1,v_2,q),dΩ)
  + stiffness(μ,t,(u_1,u_2,p),(v_1,v_2,q),dΩ)
  )

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

coupling((u_1,u_2,p),(v_1,v_2,q)) = a_12(μ,t,p,v_1,dΩ) + a_12(μ,t,p,v_2,dΩ)
induced_norm((u_1,u_2,p),(v_1,v_2,q)) = (
  a_11(μ,t,u_1,v_1,dΩ) + a_11(μ,t,u_1,v_2,dΩ) + a_11(μ,t,u_2,v_1,dΩ) + a_11(μ,t,u_2,v_2,dΩ)
  + ∫(v_1*u_1)dΩ + ∫(v_2*u_1)dΩ + ∫(v_1*u_2)dΩ + ∫(v_2*u_2)dΩ
  + ∫(dp*q)dΩ
)
reffe_u_1 = ReferenceFE(lagrangian,Float64,order)
reffe_u_2 = ReferenceFE(lagrangian,Float64,order)
# test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
# trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w,gμt_c])
test_u_1 = TestFESpace(model,reffe_u_1;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u_1 = TransientTrialParamFESpace(test_u_1,gμt_in_1)
test_u_2 = TestFESpace(model,reffe_u_2;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u_2 = TransientTrialParamFESpace(test_u_2,gμt_in_2)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u_1,test_u_2,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u_1,trial_u_2,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)

xh0μ(μ) = interpolate_everywhere([u0μ_1(μ),u0μ_2(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
# test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","perforated_plate")))
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","toy_mesh_h1")))

fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)
h1_l2_err = compute_error(results)

println(h1_l2_err)
save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

u = get_trial_fe_basis(test)
u1,u2,u3 = u

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
w = get_fe_basis(test_u)
p = get_trial_fe_basis(test_p)
x = get_cell_points(Ω)

# form((u,p),(v,q)) = ∫(p*(∇⋅(v)))dΩ
cf = ∇⋅(w)
cfx = cf(x)

# form((u_1,u_2,p),(v_1,v_2,q)) = ∫(p*(∇⋅(v_1)))dΩ + ∫(p*(∇⋅(v_2)))dΩ
w_1 = get_fe_basis(test_u_1)
w_2 = get_fe_basis(test_u_2)
df = ∇⋅(w_1) #+ p*(∇⋅(w_2))
dfx = df(x)
