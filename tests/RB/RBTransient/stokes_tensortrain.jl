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

norm_matrix = assemble_norm_matrix(feop)

X = assemble_matrix(induced_norm,test,test)

U,V = test,test
assem = TProductBlockSparseMatrixAssembler(U,V)
v = get_tp_fe_basis(V)
u = get_tp_trial_fe_basis(U)
dc = induced_norm(u,v)
# assemble_matrix(assem,collect_cell_matrix(U,V,induced_norm(u,v)))
u1,u2 = u
v1,v2 = v
dc1 = ∫(u1⋅v1)dΩ + ∫(∇(v1)⊙∇(u1))dΩ
dc2 = ∫(u2*v2)dΩ

reffe_u′ = ReferenceFE(lagrangian,Float64,order)
test_u′ = TestFESpace(Ω,reffe_u′;conformity=:H1,dirichlet_tags=["dirichlet"])
test′ = MultiFieldFESpace([test_u′,test_p];style=BlockMultiFieldStyle())

XX = assemble_matrix(induced_norm,test′,test′)
M

################# remove pressure #############
using Gridap.FESpaces
_norm_u((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ
_norm_mat = assemble_matrix(assem,collect_cell_matrix(U,V,_norm_u(u,v)))

_X = assemble_matrix(_norm_u,test′,test′)

_M1 = assemble_matrix((du,v)->∫(du⋅v)dΩ.measures_1d[1],test′[1].spaces_1d[1],test′[1].spaces_1d[1])
_A1 = assemble_matrix((du,v)->∫(∇(v)⊙∇(du))dΩ.measures_1d[1],test′[1].spaces_1d[1],test′[1].spaces_1d[1])

_M2 = assemble_matrix((du,v)->∫(du⋅v)dΩ.measures_1d[2],test′[1].spaces_1d[2],test′[1].spaces_1d[2])
_A2 = assemble_matrix((du,v)->∫(∇(v)⊙∇(du))dΩ.measures_1d[2],test′[1].spaces_1d[2],test′[1].spaces_1d[2])

_XX = kron(_M2,_M1) + kron(_A2,_M1) + kron(_M2,_A1)

_m_u((du,dp),(v,q)) = ∫(∇(v)⊙∇(du))dΩ + ∫(du⋅v)dΩ
_M = assemble_matrix(assem,collect_cell_matrix(U,V,_m_u(u,v)))

_Mok = assemble_matrix(_m_u,test′,test′)

dc1 = ∫(∇(v1)⊙∇(u1))dΩ
dc2 = ∫(v1⊙u1)dΩ
dc = dc1 + dc2

p1,p2 = get_fe_basis(test_p)

v′ = get_fe_basis(test′)
u′ = get_trial_fe_basis(test′)
dc′ = _m_u(u′,v′)
