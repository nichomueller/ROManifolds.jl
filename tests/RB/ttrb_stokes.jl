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
using FillArrays

basis,reffe_args,reffe_kwargs = reffe_u
T,order = reffe_args
cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
cell_reffes_1d = map(model->ReferenceFE(model,basis,eltype(T),order;reffe_kwargs...),model.models_1d)
space = FESpace(model.model,cell_reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
spaces_1d = FEM.univariate_spaces(model,cell_reffes_1d;conformity=:H1,dirichlet_tags=["dirichlet"])
comp_to_dofs = FEM.get_comp_to_free_dofs(T,space,cell_reffe)
dof_permutation = get_dof_permutation(T,model.model,space,order,comp_to_dofs)
tp_dof_permutation = get_tp_dof_permutation(T,model.models_1d,spaces_1d,order)

function _to_ncomps(perm::AbstractArray{S,D}) where {S,D}
  ncomp = num_components(T)
  ncomp_perm = zeros(S,size(perm)...,ncomp)
  @inbounds for comp = 1:ncomp
    selectdim(ncomp_perm,D+1,comp) .= (perm.-1).*ncomp .+ comp
  end
  return ncomp_perm
end

dof_perm,dof_perms_1d = FEM._get_tp_dof_permutation(eltype(T),model.models_1d,spaces_1d,order)
ncomp_dof_perm = _to_ncomps(dof_perm)
ncomp_dof_perms_1d = _to_ncomps.(dof_perms_1d)

simap = get_sparse_index_map(test_u,test_u)

sparsity = get_sparsity(test_u,test_u)
psparsity = FEM.permute_sparsity(sparsity,test_u,test_u)
U,V = test_u,test_u
index_map_I = get_dof_permutation(V)
index_map_J = get_dof_permutation(U)
index_map_I_1d = get_tp_dof_permutation(V).indices_1d
index_map_J_1d = get_tp_dof_permutation(U).indices_1d

psparsity = FEM.permute_sparsity(sparsity.sparsity,index_map_I,index_map_J)
