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

# fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
# results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

# println(RB.space_time_error(results))
# println(RB.speedup(results))

# save(test_dir,fesnaps)
# save(test_dir,rbop)
# save(test_dir,results)

T = Float64
N = 4
mat = soff
N_space = N-2
cores = Vector{Array{T,3}}(undef,N-1)
weights = Vector{Array{T,3}}(undef,N_space-1)
ranks = fill(1,N)
X = norm_matrix
# routine on the indexes from 1 to N_space - 1
M = RB.ttsvd_and_weights!((cores,weights,ranks),copy(mat),X;ids_range=1:FEM.get_dim(X)-1)
# routine on the indexes N_space to N_space + 1
XW = RB._get_norm_matrix_from_weights(X,weights)
# ttsvd!((cores,ranks),M,XW;ids_range=FEM.get_dim(X),kwargs...)
L,p = RB._cholesky_factor_and_perm(XW)
Ip = invperm(p)
# sizes = size(mat)
ids_range=N_space:N-1
k = ids_range[1]
Xmat = L'*reshape(_M,ranks[k]*sizes[k],:)#[p,:]
U,Σ,V = svd(Xmat)
R = RB.truncation(Σ)
core_k = (L'\U[:,1:R])[Ip,:]
ranks[k+1] = R
_M = reshape(Σ[1:R].*V[:,1:R]',R,sizes[k+1],:)
cores[k] = reshape(core_k,ranks[k],sizes[k],R)

# XW = RB._get_norm_matrix_from_weights(X,weights)
XX = RB._get_norm_matrices(X,Val(N_space))
W = weights[end]
@check length(XX) == size(W,2)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
reffe = ReferenceFE(lagrangian,Float64,2)
test = TestFESpace(model,reffe;conformity=:H1)
trial = TrialFESpace(test,x->0)
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

domain1d = (0,1)
partition1d = (2,)
model1d = CartesianDiscreteModel(domain1d,partition1d)
reffe1d = ReferenceFE(lagrangian,Float64,2)
test1d = TestFESpace(model1d,reffe1d;conformity=:H1)
trial1d = TrialFESpace(test1d,x->0)
Ω1d = Triangulation(model1d)
dΩ1d = Measure(Ω1d,2)

_model = TProductModel(domain,partition)
_test = FESpace(_model,reffe;conformity=:H1)
_perm = get_tp_dof_permutation(Float64,_model.models_1d,_test.spaces_1d,2)
perm = get_dof_permutation(Float64,model,test,2)
invp = invperm(perm[:])

# test 1
F = assemble_vector(v->∫(v)dΩ,test)
F1d = assemble_vector(v->∫(v)dΩ1d,test1d)
TPF = kronecker(F1d,F1d)
TPF ≈ F
TPF[_perm[:]] ≈ F[perm[:]]
TPF[_perm[:]][invp] ≈ F

# test 2
A = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩ,trial,test)
M1d = assemble_matrix((u,v)->∫(v*u)dΩ1d,trial1d,test1d)
A1d = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩ1d,trial1d,test1d)
TPA = kronecker(A1d,M1d) + kronecker(M1d,A1d)
TPA ≈ A
TPA[_perm[:],_perm[:]] ≈ A[perm[:],perm[:]]
TPA[_perm[:],_perm[:]][invp,invp] ≈ A


# dof perm
basis,reffe_args,reffe_kwargs = reffe
T,order = reffe_args
cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
cell_reffes_1d = map(model->ReferenceFE(model,basis,T,order;reffe_kwargs...),model.models_1d)
space = FESpace(model.model,cell_reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
spaces_1d = FEM.univariate_spaces(model,cell_reffes_1d;conformity=:H1,dirichlet_tags=["dirichlet"])
dof_permutation = get_dof_permutation(T,model.model,space,order)

spaces = spaces_1d
models = model.models_1d
D = length(models)
function _tensor_product(aprev::AbstractArray{Tp,M},a::AbstractVector{Td}) where {Tp,Td,M}
  T = promote_type(Tp,Td)
  N = M+1
  s = (size(aprev)...,length(a))
  atp = zeros(T,s)
  slicesN = eachslice(atp,dims=N)
  @inbounds for (iN,sliceN) in enumerate(slicesN)
    sliceN .= aprev .+ a[iN]
  end
  return atp
end

model_d = models[1]
space_d = spaces[1]
ndofs_d = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
ndofs = ndofs_d
cell_ids_d = get_cell_dof_ids(space_d)
dof_permutations_1d = FEM._get_dof_permutation(model_d,cell_ids_d,order)

d = 2
ndofs_prev = ndofs
node2dof_prev = dof_permutations_1d

model_d = models[d]
space_d = spaces[d]
ndofs_d = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
ndofs = ndofs_prev*ndofs_d
cell_ids_d = get_cell_dof_ids(space_d)

dof_permutations_1d = FEM._get_dof_permutation(model_d,cell_ids_d,order)

add_dim = ndofs_prev .* collect(0:ndofs_d)
add_dim_reorder = add_dim[dof_permutations_1d]
node2dof_d = _tensor_product(node2dof_prev,add_dim_reorder)
