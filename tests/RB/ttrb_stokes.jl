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
using SparseArrays
using LinearAlgebra
using Kronecker
using Mabla.FEM
using Mabla.RB

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

inflow(μ,t) = 1-cos(2π*t/(μ[1]*tf))
g_in(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_w(x,μ,t) = VectorValue(0.0,0.0)
g_w(μ,t) = x->g_w(x,μ,t)
gμt_w(μ,t) = TransientParamFunction(g_w,μ,t)
g_c(x,μ,t) = VectorValue(0.0,0.0)
g_c(μ,t) = x->g_c(x,μ,t)
gμt_c(μ,t) = TransientParamFunction(g_c,μ,t)

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
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u = TransientTrialParamFESpace(test_u,gμt_in)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ,RB.SpaceTimeMDEIM();nsnaps_state=10,nsnaps_test=5,nsnaps_mdeim=5)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","toy_mesh_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,xh0μ)

############# maps

simap = get_sparse_index_map(test_u,test_u)

U,V = test_u,test_u
sparsity = get_sparsity(U,V)
psparsity = FEM.permute_sparsity(sparsity,U,V)
I,J,_ = findnz(psparsity)
i,j,_ = FEM.univariate_findnz(psparsity)
pg2l = _global_2_local_nnz(psparsity,I,J,i,j)

IJ = get_nonzero_indices(sparsity)
lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

unrows = FEM.univariate_num_rows(sparsity)
uncols = FEM.univariate_num_cols(sparsity)
unnz = FEM.univariate_nnz(sparsity)
g2l = zeros(eltype(IJ),unnz...)

@inbounds for (k,gid) = enumerate(IJ)
  println(k)
  irows = Tuple(tensorize_indices(I[k],unrows))
  icols = Tuple(tensorize_indices(J[k],uncols))
  iaxes = CartesianIndex.(irows,icols)
  global2local = map((i,j) -> findfirst(i.==[j]),lids,iaxes)
  g2l[global2local...] = gid
end

onesparsity = SparsityPatternCSC(Mx)

V = test_u
index_map_I = FEM.get_component(get_dof_permutation(V),1)
index_map_J = get_dof_permutation(U)
index_map_I_1d = get_tp_dof_permutation(V).indices_1d
index_map_J_1d = get_tp_dof_permutation(U).indices_1d
permute_sparsity(s,(index_map_I,index_map_I_1d),(index_map_J,index_map_J_1d))

M = assemble_matrix((u,v)->∫(u⋅v)dΩ.measure,test_u.space,test_u.space)
MII = M[index_map_I[:],index_map_I[:]]

_reffe_u = ReferenceFE(lagrangian,Float64,order)
_test_u = TestFESpace(model,_reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
_M = assemble_matrix((u,v)->∫(u*v)dΩ.measure,_test_u.space,_test_u.space)
_index_map_I = get_dof_permutation(_test_u)
_MII = _M[_index_map_I[:],_index_map_I[:]]

_Mx = assemble_matrix((u,v)->∫(u*v)dΩ.measures_1d[1],_test_u.spaces_1d[1],_test_u.spaces_1d[1])
_My = assemble_matrix((u,v)->∫(u*v)dΩ.measures_1d[2],_test_u.spaces_1d[2],_test_u.spaces_1d[2])
_Mxy = kron(_My,_Mx)
