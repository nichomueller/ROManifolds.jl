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
using Mabla.FEM.ParamUtils

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

n = 30
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
g_in_1(x,μ,t) = -x[2]*(1-x[2])*inflow(μ,t)
g_in_1(μ,t) = x->g_in_1(x,μ,t)
gμt_in_1(μ,t) = TransientParamFunction(g_in_1,μ,t)
g_in_2(x,μ,t) = 0.0
g_in_2(μ,t) = x->g_in_2(x,μ,t)
gμt_in_2(μ,t) = TransientParamFunction(g_in_2,μ,t)

u0(x,μ) = 0.0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

stiffness(μ,t,(u1,u2,p),(v1,v2,q),dΩ) = (∫(aμt(μ,t)*∇(v1)⋅∇(u1))dΩ + ∫(aμt(μ,t)*∇(v2)⋅∇(u2))dΩ
  - ∫(p*∂ₓ₁(v1))dΩ - ∫(p*∂ₓ₂(v2))dΩ + ∫(q*∂ₓ₁(u1))dΩ + ∫(q*∂ₓ₂(u2))dΩ)
mass(μ,t,(u1ₜ,u2ₜ,pₜ),(v1,v2,q),dΩ) = ∫(v1*u1ₜ)dΩ + ∫(v2*u2ₜ)dΩ
res(μ,t,(u1,u2,p),(v1,v2,q),dΩ) = ∫(v1*∂t(u1))dΩ + ∫(v2*∂t(u2))dΩ + stiffness(μ,t,(u1,u2,p),(v1,v2,q),dΩ)

trian_res = (Ω.trian,)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

coupling((u1,u2,p),(v1,v2,q)) = ∫(p*∂ₓ₁(v1))dΩ + ∫(p*∂ₓ₂(v2))dΩ
induced_norm((u1,u2,p),(v1,v2,q)) = ∫(v1*u1)dΩ + ∫(v2*u2)dΩ + ∫(∇(v1)⊙∇(u1))dΩ + ∫(∇(v2)⊙∇(u2))dΩ + ∫(p*q)dΩ

reffe_u = ReferenceFE(lagrangian,Float64,order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u1 = TransientTrialParamFESpace(test_u,gμt_in_1)
trial_u2 = TransientTrialParamFESpace(test_u,gμt_in_2)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:C0)
# test_p = TestFESpace(Ω,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u1,trial_u2,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","tensor_train")))

# fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
# results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)
fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,rbstats,rbstats)

compute_error(results)

# comparison with standard TT-SVD
sTT = fesnaps
X = assemble_norm_matrix(feop)
@time ttsvd(sTT[1],X[1])
basis = Projection(sTT,X)
b1 = basis[1]
cores1 = get_cores(b1)
RBTransient.check_orthogonality(cores1,X[1])

#supr
@time B = assemble_coupling_matrix(feop)
@time basis′ = enrich_basis(basis,X,B)
b1′ = basis′[1]
cores1′ = get_cores(b1′)
RBTransient.check_orthogonality(cores1′,X[1])

#MDEIM
using Gridap.CellData
using Gridap.FESpaces
using Gridap.ODEs

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
op = get_algebraic_operator(feop)
op = TransientPGOperator(op,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RBSteady.mdeim_params(rbsolver))
j,r = jacobian_and_residual(rbsolver,op,smdeim)
red_jac = reduced_jacobian(rbsolver,op,j)
red_res = reduced_residual(rbsolver,op,r)

j1 = j[1][1]
trian = get_domains(j1)[1]
basis = reduced_basis(j1[1,1];randomized=true)
lu_interp,integration_domain = mdeim(rbsolver.mdeim_style,basis)
red_trial1,red_test1 = red_trial[1],red_test[1]
proj_basis = reduce_operator(rbsolver.mdeim_style,basis,red_trial1,red_test1)

mat_k = reshape(j1[1,1],size(j1[1,1],1),:)
Ur,Σr,Vr = RBSteady._tpod(mat_k;randomized=true)

# reduced_form(rbsolver,j[1,1],trian,trial[j],test[i];kwargs...)

# TPOD
using Mabla.FEM.IndexMaps
using BlockArrays
using SparseArrays
using LinearAlgebra
_dΩ = dΩ.measure

_coupling((u1,u2,p),(v1,v2,q)) = ∫(p*∂ₓ₁(v1))_dΩ + ∫(p*∂ₓ₂(v2))_dΩ
_induced_norm((u1,u2,p),(v1,v2,q)) = ∫(v1*u1)_dΩ + ∫(v2*u2)_dΩ + ∫(∇(v1)⊙∇(u1))_dΩ + ∫(∇(v2)⊙∇(u2))_dΩ + ∫(p*q)_dΩ

_test_u = TestFESpace(model.model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
_trial_u1 = TransientTrialParamFESpace(_test_u,gμt_in_1)
_trial_u2 = TransientTrialParamFESpace(_test_u,gμt_in_2)
_test_p = TestFESpace(model.model,reffe_p;conformity=:C0)
_trial_p = TrialFESpace(_test_p)
_test = TransientMultiFieldParamFESpace([_test_u,_test_u,test_p];style=BlockMultiFieldStyle())
_trial = TransientMultiFieldParamFESpace([_trial_u1,_trial_u2,_trial_p];style=BlockMultiFieldStyle())
_feop = TransientParamLinearFEOperator((stiffness,mass),res,_induced_norm,ptspace,
  _trial,_test,_coupling,trian_res,trian_stiffness,trian_mass)

# basis (comp 1 velocity)
_s = change_index_map(TrivialIndexMap,fesnaps)
@time _X = assemble_norm_matrix(_feop)
@time Projection(_s[1],_X[Block(1,1)])
@time _basis = Projection(_s,_X)

@time _B = assemble_coupling_matrix(_feop)
@time _basis′ = enrich_basis(_basis,_X,_B)

#MDEIM
_red_trial,_red_test = reduced_fe_space(rbsolver,_feop,_s)
_op = get_algebraic_operator(_feop)
_op = TransientPGOperator(_op,_red_trial,_red_test)
_smdeim = select_snapshots(_s,RBSteady.mdeim_params(rbsolver))
_j,_r = jacobian_and_residual(rbsolver,_op,_smdeim)

_j1 = _j[1][1]
_trian = get_domains(_j[1])[1]
_basis = reduced_basis(_j1[1,1])
_lu_interp,_integration_domain = mdeim(rbsolver.mdeim_style,_basis)
_red_trial1,_red_test1 = _red_trial[1],_red_test[1]
_proj_basis = reduce_operator(rbsolver.mdeim_style,_basis,_red_trial1,_red_test1)

_mat = flatten_snapshots(_j1[1,1])
_Ur,_Σr,_Vr = RBSteady._tpod(_mat;randomized=true)


# product of reshape
sizes = size(sTT[1])
mat_k = reshape(sTT[1],sizes[1],:)
@time RBSteady._tpod(mat_k)

mat_k′ = Base.ReshapedArray(mat_k.snaps,mat_k.size,mat_k.mi)
@time RBSteady._tpod(mat_k′)
