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
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","toy_mesh_h1")))

# fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)
# save(test_dir,fesnaps)
fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))

s = fesnaps
soff = select_snapshots(s,RBSteady.offline_params(rbsolver))
norm_matrix = assemble_norm_matrix(feop)
basis = reduced_basis(soff,norm_matrix)

############### enrichment ###############
using BlockArrays

supr_op = assemble_coupling_matrix(feop)
pblocks,dblocks = TProduct.primal_dual_blocks(supr_op)
cores_space = get_cores_space(basis)
core_time = RBTransient.get_core_time(basis)
norms_primal = map(ip -> norm_matrix[Block(ip,ip)],pblocks)
cores_primal = map(ip -> BlockVectorTTCores.([cores_space[ip]...,core_time[ip]]),pblocks)

T = Float64

id = dblocks[1]
rcores = Vector{Array{T,3}}[]
rcore = Matrix{T}[]
cores_dual_space_i = cores_space[id]
core_dual_time_i = core_time[id]
for ip in pblocks
  A = norm_matrix[Block(ip,ip)]
  C = supr_op[Block(ip,id)]
  cores_primal_space_i = cores_space[ip]
  core_primal_time_i = core_time[ip]
  RBSteady.reduced_coupling!((rcores,rcore),cores_primal_space_i,core_primal_time_i,
    cores_dual_space_i,core_dual_time_i,A,C)
end
# enrich!(cores_primal′,rcores,vcat(rcore...))
rcore = vcat(rcore...)

flag=false
# suppose we have to update with i = 10
i = 10

# flag = RBSteady._push_to_primal!(cores_primal[ip],rcores[ip],norms_primal[ip],i;flag)

D = length(cores_primal[ip])
T = eltype(eltype(first(cores_primal[ip])))
weights = Vector{Array{T,3}}(undef,D-1)

for d in 1:D-1
  push!(cores_primal[ip][d],rcores[ip][d])
  RBSteady._weight_array!(weights,cores_primal[ip][d],norms_primal[ip],Val{d}())
end

push!(cores_primal[ip][D],rcores[ip][D][:,:,i:i])
# RBSteady.orthogonalize!(cores_primal[ip][D],norms_primal[ip])
N_space = length(weights)
X = norms_primal[ip]
XW = RBSteady._get_norm_matrix_from_weights(X,weights)
L,p = RBSteady._cholesky_factor_and_perm(XW)
cN = cores_primal[ip][D][N_space]
mat = reshape(cN,:,size(cN,3))
XWmat = L'*mat[p,:]
Q̃,R = qr(XWmat)
cores[N_space] = reshape((L'\Q̃)[invperm(p),:],size(cN))

##########################################

using Gridap.CellData
using Gridap.FESpaces

smdeim = select_snapshots(fesnaps,RBSteady.mdeim_params(rbsolver))
rbtrial,rbtest = fe_subspace(trial,basis),fe_subspace(test,basis)
op = TransientPGOperator(get_algebraic_operator(feop),rbtrial,rbtest)
jjac,rres = jacobian_and_residual(rbsolver,op,smdeim)
red_jac = reduced_jacobian(rbsolver,op,jjac)
red_res = reduced_residual(rbsolver,op,rres)
trians_rhs = get_domains(red_res)
trians_lhs = map(get_domains,red_jac)
new_op = change_triangulation(op,trians_rhs,trians_lhs)
rbop = TransientPGMDEIMOperator(new_op,red_jac,red_res)

# ######
# using Mabla.FEM.IndexMaps
# s = jjac[1][1][1,3]
# basis = Projection(s)

# cores_space...,core_time = ttsvd(s)
# cores_space′ = recast(s,cores_space)
# index_map = get_index_map(s)
# bs = RBSteady._cores2basis(index_map,cores_space′[1],cores_space′[2])


# basis_spacetime = RBTransient.get_basis_spacetime(basis)
# indices_spacetime,interp_basis_spacetime = empirical_interpolation(basis_spacetime)
# indices_space = fast_index(indices_spacetime,num_space_dofs(basis))
# indices_time = slow_index(indices_spacetime,num_space_dofs(basis))
# lu_interp = lu(interp_basis_spacetime)
# integration_domain = TransientIntegrationDomain(indices_space,indices_time)
# ######

rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)
println(compute_error(results))
println(compute_speedup(results))

using BlockArrays
compute_error(results.sol[1],results.sol_approx[1],results.norm_matrix[Block(1,1)])
compute_error(results.sol[2],results.sol_approx[2],results.norm_matrix[Block(2,2)])
compute_error(results.sol[3],results.sol_approx[3],results.norm_matrix[Block(3,3)])
