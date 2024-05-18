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
using Kronecker
using Mabla.FEM
using Mabla.RB

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.05

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (5,5)
model = TProductModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=2)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)

boh = BasicSnapshots(fesnaps)
bohI = select_snapshots(fesnaps,1:5;spacerange=(1:2,1:3))

# red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# odeop = get_algebraic_operator(feop)
# pop = PODOperator(odeop,trial,test)
# smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
# A,b = jacobian_and_residual(rbsolver,pop,smdeim)
soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
norm_matrix = assemble_norm_matrix(feop)
YE = ttsvd(soff,norm_matrix)

# T,N,mat,X = Float64,4,soff,norm_matrix
# N_space = N-2
# cores = Vector{Array{T,3}}(undef,N-1)
# weights = Vector{Array{T,3}}(undef,N_space-1)
# ranks = fill(1,N)
# sizes = size(mat)

# for k in 1:FEM.get_dim(X)-1
#   mat_k = reshape(mat,ranks[k]*sizes[k],:)
#   U,Σ,V = svd(mat_k)
#   rank = RB.truncation(Σ)
#   core_k = U[:,1:rank]
#   ranks[k+1] = rank
#   mat = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
#   cores[k] = reshape(core_k,ranks[k],sizes[k],rank)
#   RB._weight_array!(weights,cores,X,Val(k))
# end
# XW = RB._get_norm_matrix_from_weights(X,weights)

# BOH

# fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
# results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

# println(RB.space_time_error(results))
# println(RB.speedup(results))

# save(test_dir,fesnaps)
# save(test_dir,rbop)
# save(test_dir,results)

# vals = map(get_values,fesnaps.values)
# r = fesnaps.realization
# s = Snapshots(vals,r)
# soff1 = select_snapshots(s,RB.offline_params(rbsolver))
# using SparseArrays
# ye = ttsvd(soff,sparse(copy(norm_matrix)))

# S = copy(soff)
# norms = Vector{Vector{Matrix{Float64}}}(undef,3)
# norms[1] = [norm_matrix.arrays_1d[1],norm_matrix.arrays_1d[2]]
# norms[2] = [norm_matrix.gradients_1d[1],norm_matrix.arrays_1d[2]]
# norms[2] = [norm_matrix.arrays_1d[1],norm_matrix.gradients_1d[2]]
# resok = ttsvd_thick(S,norms,1e-4)

cores = YE
bx = cores[1][1,:,:]
corey = cores[2]
bxy = zeros(size(bx,1)*size(corey,2),size(corey,3))
for i = axes(corey,3)
  bxy[:,i] = sum([kronecker(bx[:,k],corey[k,:,i]) for k = axes(bx,2)])
end

bx'*bx
bxy'*bxy
bxy'*X*bxy

coresnew = ttsvd(soff)
bx = coresnew[1][1,:,:]
corey = coresnew[2]
bxy = zeros(size(bx,1)*size(corey,2),size(corey,3))
for i = axes(corey,3)
  bxy[:,i] = sum([kronecker(bx[:,k],corey[k,:,i]) for k = axes(bx,2)])
end
bx'*bx
bxy'*bxy
