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

test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
B = assemble_matrix((dp,v)->∫(dp*(∇⋅(v)))dΩ.measure,test_p.space,test_u.space)
m2 = feop.op.op.index_map.matrix_map[2]
B[m2]

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

# serialize(RBSteady.get_snapshots_filename(test_dir),snaps)
snaps = deserialize(RBSteady.get_snapshots_filename(test_dir))

s = snaps
soff = select_snapshots(s,RBSteady.offline_params(rbsolver))
norm_matrix = assemble_norm_matrix(feop)
basis = reduced_basis(soff,norm_matrix)
# enrich_basis(feop,basis,norm_matrix)
supr_op = assemble_coupling_matrix(feop)
enrich_basis(basis,norm_matrix,supr_op)

basis_space = get_basis_space(basis)

red_trial,red_test = reduced_fe_space(rbsolver,feop,snaps)
op = get_algebraic_operator(feop)

X11 = norm_matrix[Block(1,1)]

reffe_u′ = ReferenceFE(lagrangian,Float64,order)
test_u′ = TestFESpace(Ω,reffe_u′;conformity=:H1,dirichlet_tags=["dirichlet"])
induced_norm′(du,v) = ∫(du*v)dΩ.measure + ∫(∇(v)⋅∇(du))dΩ.measure
X = assemble_matrix(induced_norm′,test_u′.space,test_u′.space)


# #
# basis_space_1 = get_basis_space(basis[1])
# cores_space_1 = basis[1].cores_space

# dp,v = get_trial_fe_basis(test_p.spaces_1d[1]),get_fe_basis(test_u.spaces_1d[1])
# # ∫(dp*(∇⋅(v)))dΩ.measures_1d[1]
# using Gridap.CellData
# using Gridap.Arrays
# using Gridap.Fields

# x = get_cell_points(Ω.trians_1d[1])
# dv = ∇⋅(v)
# dvx = dv(x)
# px = dp(x)
# (dp*dv)(x)[1]

# assemble_matrix((dp,v)->∫(dp*v)dΩ.measures_1d[1],)

# dp2 = get_trial_fe_basis(test_p)
# v2 = get_fe_basis(test_u)
# x2 = get_cell_points(Ω.trian)
# dv2 = ∇⋅(v2)
# dvx2 = dv2(x2)
# px2 = dp2(x2)

# (dp2*dv2)(x2)[1]

# test_u′ = TestFESpace(Ω,reffe_u;conformity=:H1)
# test_p′ = TestFESpace(Ω,reffe_p;conformity=:H1)

# B1=assemble_matrix((dp,v)->∫(dp*(∇⋅(v)))dΩ.measures_1d[1],test_p′.spaces_1d[1],test_u′.spaces_1d[1])
# B2=assemble_matrix((dp,v)->∫(dp*(∇⋅(v)))dΩ.measures_1d[2],test_p′.spaces_1d[2],test_u′.spaces_1d[2])
# C1=assemble_matrix((dp,v)->∫(dp*v)dΩ.measures_1d[1],test_p′.spaces_1d[1],test_u′.spaces_1d[1])
# C2=assemble_matrix((dp,v)->∫(dp*v)dΩ.measures_1d[2],test_p′.spaces_1d[2],test_u′.spaces_1d[2])
# B̃ = kron(B2,C1)+kron(C2,B1)
# row_index_map = get_tp_dof_index_map(test_u′)
# row_index_map1 = row_index_map[:,:,1]
# B̃′ = B̃[vec(row_index_map1),:]

# B = assemble_matrix((dp,v)->∫(dp*(∇⋅(v)))dΩ.measure,test_p′.space,test_u′.space)
# imap = get_vector_index_map(feop)[1]
# imap1 = imap[:,:,1]
# B′ = B[vec(imap1),:]

n = 2
domain = (0,1,0,1)
partition = (n,n)
model = TProductModel(domain,partition)
order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

test_u′ = TestFESpace(Ω,reffe_u;conformity=:H1)
test_p′ = TestFESpace(Ω,reffe_p;conformity=:H1)

V1=assemble_vector((v)->∫((∇⋅(v)))dΩ.measures_1d[1],test_u′.spaces_1d[1])
V2=assemble_vector((v)->∫((∇⋅(v)))dΩ.measures_1d[2],test_u′.spaces_1d[2])
W1=assemble_vector((v)->∫(v)dΩ.measures_1d[1],test_u′.spaces_1d[1])
W2=assemble_vector((v)->∫(v)dΩ.measures_1d[2],test_u′.spaces_1d[2])
V′ = kron(V2,W1) + kron(W2,V1)

V = assemble_vector((v)->∫((∇⋅(v)))dΩ.measure,test_u′.space)#[1:121]

row_index_map = get_dof_index_map(test_u′)
col_index_map = get_dof_index_map(test_p′)
tp_row_index_map = get_tp_dof_index_map(test_u′)
matmap = get_matrix_index_map(test_u′,test_p′)

V[row_index_map[:]][1:25] ≈ V′[tp_row_index_map[:,:,1][:]]

using Mabla.FEM.IndexMaps
using Mabla.FEM.ParamSteady
using SparseArrays

sparsity = get_sparsity(test_p′,test_u′)
psparsity = permute_sparsity(sparsity,test_p′,test_u′)
I,J,_ = findnz(psparsity)
i,j,_ = IndexMaps.univariate_findnz(psparsity)
g2l = ParamSteady._global_2_local_nnz(psparsity,I,J,i,j)
pg2l = ParamSteady._permute_index_map(g2l,test_p′,test_u′)
SparseIndexMap(pg2l,psparsity)
