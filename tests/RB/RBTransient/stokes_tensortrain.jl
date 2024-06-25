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
model = TProductModel(domain, partition)
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
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u = TransientTrialParamFESpace(test_u,gμt_in)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
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


TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)


i = feop.op.op.index_map
imat = i.matrix_map

using Mabla.FEM.IndexMaps
using SparseArrays

# index_map = FEOperatorIndexMap(trial,test)
# get_matrix_index_map(trial[1].space,test[2])
U,V = trial[1].space,test[2]
sparsity = get_sparsity(U,V)
psparsity = permute_sparsity(sparsity,U,V)
I,J,_ = findnz(psparsity)
i,j,_ = IndexMaps.univariate_findnz(psparsity)
g2l = ParamSteady._global_2_local_nnz(psparsity,I,J,i,j)
# pg2l = ParamSteady._permute_index_map(g2l,U,V)
I = get_dof_index_map(V)
J = get_dof_index_map(U)
nrows = num_free_dofs(V)
# ParamSteady._permute_index_map(index_map,I,J,nrows)

function _to_component_indices(i,ncomps,icomp,nrows)
  ic = copy(i)
  @inbounds for (j,IJ) in enumerate(i)
    IJ == 0 && continue
    I = fast_index(IJ,nrows)
    J = slow_index(IJ,nrows)
    J′ = (J-1)*ncomps + icomp
    ic[j] = (J′-1)*nrows + I
  end
  return ic
end

ncomps = num_components(J)

J1 = get_component(J,1;multivalue=false)
index_map′ = ParamSteady._permute_index_map(g2l,I,J1,nrows)
index_map′′ =  _to_component_indices(index_map′,ncomps,1,nrows)


using SparseArrays
struct MyRandStruct{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
  a::SparseMatrixCSC{Tv,Ti}
  b
end

Base.size(a::MyRandStruct) = size(a.b)
function Base.getindex(a::MyRandStruct,i,j)
  if a.b[i,j] == 0
    return zero(eltype(a))
  end
  a.a[a.b[i,j]]
end

B = assemble_matrix((u,q) -> ∫(q*(∇⋅(u)))dΩ.measure,trial_u(nothing),test_p)
b = MyRandStruct(B,imat[3][:,:,1])

A = assemble_matrix((u,q) -> ∫(∇(q)⊙∇(u))dΩ.measure,trial_u(nothing),test_u)
A[imat[1][:,:,2]]

U,V = test[2],test[1]
sparsity = get_sparsity(U,V)
psparsity = permute_sparsity(sparsity,U,V)
I,J,_ = findnz(psparsity)
i,j,_ = IndexMaps.univariate_findnz(psparsity)
g2l = ParamSteady._global_2_local_nnz(psparsity,I,J,i,j)
I = get_dof_index_map(V)
J = get_dof_index_map(U)
nrows = num_free_dofs(V)

function _to_component_indices(i,ncomps,icomp,nrows)
  ic = copy(i)
  @inbounds for (j,IJ) in enumerate(i)
    IJ == 0 && continue
    I = fast_index(IJ,nrows)
    J = slow_index(IJ,nrows)
    I′ = (I-1)*ncomps + icomp
    ic[j] = (J-1)*nrows*ncomps + I′
  end
  return ic
end

ncomps = num_components(I)
nrows_per_comp = Int(nrows/ncomps)

I1 = get_component(I,1;multivalue=false)
index_map′ = ParamSteady._permute_index_map(g2l,I1,J,nrows_per_comp)
index_map′′ = _to_component_indices(index_map′,ncomps,1,nrows_per_comp)

# test with A
UA,VA = test[1],test[1]
sparsityA = get_sparsity(UA,VA)
psparsityA = permute_sparsity(sparsityA,UA,VA)
IA,JA,_ = findnz(psparsityA)
iA,jA,_ = IndexMaps.univariate_findnz(psparsityA)
g2lA = ParamSteady._global_2_local_nnz(psparsityA,IA,JA,iA,jA)
IA = get_dof_index_map(VA)
JA = get_dof_index_map(UA)
nrowsA = num_free_dofs(VA)

function _to_component_indices_A(i,ncomps,icomp,nrows)
  ic = copy(i)
  @inbounds for (j,IJ) in enumerate(i)
    IJ == 0 && continue
    I = fast_index(IJ,nrows)
    J = slow_index(IJ,nrows)
    I′ = (I-1)*ncomps + icomp
    J′ = (J-1)*ncomps + icomp
    ic[j] = (J′-1)*nrows*ncomps + I′
  end
  return ic
end

ncomps_IA = num_components(IA)
ncomps = ncomps_IA
nrows_per_compA = Int(nrowsA/ncomps)

I1A = get_component(IA,1;multivalue=false)
J1A = get_component(JA,1;multivalue=false)
index_map′A = ParamSteady._permute_index_map(g2lA,I1A,J1A,nrows_per_compA)

index_map′′A = _to_component_indices_A(index_map′A,ncomps,1,nrows_per_compA)

i1 = fast_index(1912,nrows_per_compA)
j1 = slow_index(1912,nrows_per_compA)
i1′ = (i1-1)*ncomps + 1
j1′ = (j1-1)*ncomps + 1
(j1′-1)*nrows_per_compA*ncomps + i1′

# assemble B with order 1
reffe_u1 = ReferenceFE(lagrangian,VectorValue{2,Float64},1)
test_u1 = TestFESpace(model,reffe_u1;conformity=:H1,dirichlet_tags=["dirichlet"])
B1 = assemble_matrix((u,q) -> ∫(q*(∇⋅(u)))dΩ.measure,test_u1.space,test_p.space)

U,V = test_p,test_u1
sparsity = get_sparsity(U,V)
psparsity = permute_sparsity(sparsity,U,V)
I,J,_ = findnz(psparsity)
i,j,_ = IndexMaps.univariate_findnz(psparsity)
g2l = ParamSteady._global_2_local_nnz(psparsity,I,J,i,j)
I = get_dof_index_map(V)
J = get_dof_index_map(U)
nrows = num_free_dofs(V)

ncomps = num_components(I)
nrows_per_comp = Int(nrows/ncomps)

I1 = get_component(I,1;multivalue=false)
index_map′ = ParamSteady._permute_index_map(g2l,I1,J,nrows_per_comp)
index_map′′ = _to_component_indices(index_map′,ncomps,1,nrows_per_comp)

# ok
boh = MyRandStruct(B1,index_map′′)
rr = fill!(psparsity.sparsity.matrix.nzval,1.0)
boh = MyRandStruct(psparsity.sparsity.matrix,g2l)

index_map_I = get_dof_index_map(V)
index_map_J = get_dof_index_map(U)
# index_map_I_1d = get_tp_dof_index_map(V).indices_1d
# index_map_J_1d = get_tp_dof_index_map(U).indices_1d
# permute_sparsity(sparsity,(index_map_I,index_map_I_1d),(index_map_J,index_map_J_1d))
# permute_sparsity(sparsity.sparsity,index_map_I,index_map_J)
ids1 = get_component(index_map_I,1;multivalue=false)
spars = permute_sparsity(sparsity.sparsity,ids1,index_map_J)
