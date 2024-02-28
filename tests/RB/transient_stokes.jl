using Gridap
using Test
using DrWatson
using Gridap.MultiField
using Mabla.FEM
using Mabla.RB

θ = 1
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

inflow(μ,t) = 1-cos(2π*t/tf)+sin(μ[2]*2π*t/tf)/μ[1]
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

res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
jac(μ,t,u,(du,dp),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ + ∫(q*(∇⋅(du)))dΩ
jac_t(μ,t,u,(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
# test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
# trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w,gμt_c])
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u = TransientTrialParamFESpace(test_u,gμt_in)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test,coupling)
feop = FEOperatorWithTrian(_feop,trian_res,trian_jac,trian_jac_t)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ,RB.SpaceTimeMDEIM();nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
# test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","perforated_plate")))
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","toy_mesh_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(feop,rbsolver,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ,RB.SpaceOnlyMDEIM();nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","perforated_plate")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver_space,dir=test_dir_space)

rbop_space = reduced_operator(rbsolver_space,feop,fesnaps)
rbsnaps_space,rbstats_space = solve(rbsolver_space,rbop,fesnaps)
results_space = rb_results(feop,rbsolver_space,fesnaps,rbsnaps_space,festats,rbstats_space)

println(RB.space_time_error(results_space))
save(test_dir,rbop_space)
save(test_dir,results_space)

# reduced_fe_space(rbsolver,feop,fesnaps)
soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
norm_matrix = assemble_norm_matrix(feop)
# bases = reduced_basis(feop,soff,norm_matrix;ϵ=RB.get_tol(rbsolver))
bases = reduced_basis(feop.op.op,soff,norm_matrix;ϵ=RB.get_tol(rbsolver))
ebases = RB.enrich_basis(feop,bases,norm_matrix)

# RB.space_time_error(results)
s1 = soff[1]
basis_space,basis_time = bases
ebasis_space,ebasis_time = ebases

using BlockArrays
bs1 = basis_space[1]
bt1 = basis_time[1]
ebs1 = ebasis_space[1]
ebt1 = ebasis_time[1]

es1 = norm(s1 - bs1*bs1'*s1) / norm(s1)
ees1 = norm(s1 - ebs1*ebs1'*norm_matrix[Block(1,1)]*s1) / norm(s1)

s2 = soff[2]

using LinearAlgebra
B = assemble_coupling_matrix(feop)
A = norm_matrix[Block(1,1)]
Chol = cholesky(A)
basis_primal,basis_dual = basis_space.array
b_i = B[Block(1,2)] * basis_dual
supr_i = Chol \ b_i
S = copy(supr_i)
gram_schmidt!(S,basis_primal,A)

S'*A*S
BBB = hcat(basis_primal,S)

BBB'*A*BBB

# reduced_basis(feop.op.op,soff,norm_matrix;ϵ=RB.get_tol(rbsolver))
# reduced_basis(soff,norm_matrix;ϵ=RB.get_tol(rbsolver))
