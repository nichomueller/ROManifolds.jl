using Gridap
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
using Mabla.RB.RBTransient

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 5
domain = (0,1,0,1)
partition = (n,n)
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = 1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = 1-cos(2π*t/tf)+sin(μ[2]*2π*t/tf)/μ[1]
g_in(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u = TransientTrialParamFESpace(test_u,gμt_in)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:H1,constraint=:zeromean)
# test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","toy_mesh_h1")))

# fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)

nparams = num_params(rbsolver)
sol = solve(fesolver,feop,xh0μ;nparams)
odesol = sol.odesol
r = odesol.r
stats = @timed begin
  vals = collect(odesol)
end
i = get_vector_index_map(feop)
snaps = Snapshots(vals,i,r)

s = snaps
soff = select_snapshots(s,RBSteady.offline_params(rbsolver))
# norm_matrix = assemble_norm_matrix(feop)
V = test
U = evaluate(trial,nothing)
assem = TProductBlockSparseMatrixAssembler(U,V)
v = get_tp_fe_basis(V)
u = get_tp_trial_fe_basis(U)

# induced_norm((u[1],u[2]),(v[1],v[1]))
using Gridap.Arrays
using Gridap.CellData

∫(u[1]⋅v[1])dΩ
# u[1]⋅v[1]
c = return_cache(Operation(⋅),u[1],v[1])
# evaluate!(c,Operation(⋅),u[1],v[1])
evaluate!(c[1],Operation(⋅),get_data(u[1])[1],get_data(v[1])[1])


v′ = get_fe_basis(V)
u′ = get_trial_fe_basis(U)
# u′[1]⋅v′[1]
c′ = return_cache(Operation(⋅),u′[1],v′[1])
evaluate!(c′,Operation(⋅),u′[1],v′[1])

# m1 = CartesianDiscreteModel((0,1),(5,))
# Ω1 = Triangulation(m1)
# dΩ1 = Measure(Ω1,degree)
# t1 = TestFESpace(m1,reffe_u;conformity=:H1)
# t2 = TestFESpace(m1,reffe_p;conformity=:C0)

# v1 = get_fe_basis(t1)
# u1 = get_trial_fe_basis(t1)
# v2 = get_fe_basis(t2)
# u2 = get_trial_fe_basis(t2)

# _test_u = TestFESpace(Ω,reffe_u;conformity=:H1)
# assemble_matrix((u,v)->∫(u⋅v)dΩ.measure,_test_u.space,_test_u.space)
# assemble_matrix((u,v)->∫(u⋅v)dΩ1,t1,t1)
# assemble_matrix((u,v)->∫(v*(∇⋅(u)))dΩ1,t1,t2)
# cf = ∇⋅(u1)
# cf(get_cell_points(Ω1))
