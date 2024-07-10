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

trian_res = (Ω.trian,)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

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

# nparams = num_params(rbsolver)
# sol = solve(fesolver,feop,xh0μ;nparams)
# odesol = sol.odesol
# r = odesol.r
# stats = @timed begin
#   vals = collect(odesol)
# end
# i = get_vector_index_map(feop)
# snaps = Snapshots(vals,i,r)

# serialize(RBSteady.get_snapshots_filename(test_dir),snaps)
snaps = deserialize(RBSteady.get_snapshots_filename(test_dir))

s = snaps
soff = select_snapshots(s,RBSteady.offline_params(rbsolver))
norm_matrix = assemble_norm_matrix(feop)
# supr_op = assemble_coupling_matrix(feop)
basis = reduced_basis(soff,norm_matrix)

# using Kronecker
# bs_primal,bs_dual... = add_space_supremizers(basis,norm_matrix,supr_op)
# bt_primal,bt_dual... = add_time_supremizers(basis)
# bst_primal = kronecker(bt_primal,bs_primal)
# snaps_primal = RBSteady.basis2cores(bst_primal,basis[1,1])
# cores_space_primal...,core_time_primal = ttsvd(snaps_primal,norm_matrix[1,1];ϵ=1e-10)
# _,cores_space_dual... = get_cores_space(basis).array
# _,core_time_dual... = get_cores_time(basis).array
# cores_space = (cores_space_primal,cores_space_dual...)
# cores_time = (core_time_primal,core_time_dual...)
# basis = BlockProjection(map(TransientTTSVDCores,cores_space,cores_time),basis.touched)

# # test for X
# using BlockArrays
# using LinearAlgebra
# test′ = MultiFieldFESpace([test_u.space,test_p.space];style=BlockMultiFieldStyle())
# X = assemble_matrix(induced_norm,test′,test′)
# X1 = X[Block(1,1)]
# imap = feop.op.op.index_map.matrix_map
# imap1 = imap[1,1]
# X1nnz = Matrix(X1[imap1[:,:,1]])
# U,S,V = svd(X1nnz)
# ttsvd(X1nnz)

# MDEIM
using Gridap.FESpaces
smdeim = select_snapshots(snaps,RBSteady.mdeim_params(rbsolver))
rbtrial,rbtest = fe_subspace(trial,basis),fe_subspace(test,basis)
op = TransientPGOperator(get_algebraic_operator(feop),rbtrial,rbtest)
jjac,rres = jacobian_and_residual(rbsolver,op,smdeim)
# red_jac = reduced_jacobian(rbsolver,op,jjac)
# red_res = reduced_residual(rbsolver,op,rres)

using Kronecker
using Mabla.FEM.IndexMaps
A1 = jjac[1][1][2,1]
basis = reduced_basis(A1)
lu_interp,integration_domain = mdeim(rbsolver.mdeim_style,basis)
combine = (x,y) -> θ*x+(1-θ)*y
proj_basis = reduce_operator(rbsolver.mdeim_style,basis,rbtrial[1],rbtest[2];combine)

aa,bb,cc = basis.cores_space[1],rbtrial[1].basis.cores_space[1],rbtest[2].basis.cores_space[1]
aa,bb,cc = basis.cores_space[2],rbtrial[1].basis.cores_space[2],rbtest[2].basis.cores_space[2]
aa,bb,cc = basis.cores_space[3],rbtrial[1].basis.cores_space[3],rbtest[2].basis.cores_space[3]
compress_core(aa,bb,cc)


# red_trian = reduce_triangulation(Ω.trian,integration_domain,rbtrial[2,1],rbtest[2,1])
# coefficient = allocate_coefficient(rbsolver,basis)
# result = allocate_result(rbsolver,rbtrial[2,1],rbtest[2,1])
# ad = AffineDecomposition(proj_basis,lu_interp,integration_domain,coefficient,result)

bform(dp,v) = ∫(dp*(∇⋅(v)))dΩ.measure
aform(du,v) = ∫(du⋅v)dΩ.measure

B = assemble_matrix(bform,test_p.space,test_u.space)
A = assemble_matrix(aform,test_u.space,test_u.space)

imapA_row = feop.op.op.index_map.vector_map[1]
imapA_col = imapA_row
A[imapA_row[:,:,1][:],imapA_row[:,:,1][:]]

imapA = feop.op.op.index_map.matrix_map[1]
A[imapA[:,:,1]]
