using Gridap
using Test
using DrWatson
using Serialization

using Mabla.FEM
using Mabla.FEM.Utils
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

# time marching
θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 0.3

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# geometry
n = 5
domain = (0,1,0,1)
partition = (n,n)
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet","boundary")

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

# weak formulation
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
rhs(μ,t,v,dΩ) = ∫(fμt(μ,t)*v)dΩ
res(μ,t,u,v,dΩ) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ)

trian_res = (Ω.trian,)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

energy(du,v) = ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)

tol = fill(1e-4,4)
state_reduction = TTSVDReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_test=5,nparams_res=30,nparams_jac=20)
test_dir = datadir(joinpath("heateq","test_tt_$(1e-4)"))
create_dir(test_dir)

fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ;r)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats,cache = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(results)

save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

soff = select_snapshots(fesnaps,RBSteady.offline_params(rbsolver))
X = assemble_matrix(feop,energy)
# projection(state_reduction,soff,X)
red_style = ReductionStyle(state_reduction)
cores,remainder = ttsvd(red_style,soff,X)
core_t,remainder_t = RBSteady.ttsvd_loop(red_style[ndims(soff)-1],remainder)
push!(cores,core_t)

# # cores_k,remainders_k = map(k -> ttsvd(red_style,soff,X[k]),1:rank(X)) |> tuple_of_arrays

A = reshape(soff,1,size(soff,1),:)
X = X[1]
core_d,remainder_d = RBSteady.ttsvd_loop(red_style[1],A,X[1])
oldrank = size(core_d,3)
A = reshape(remainder_d,oldrank,size(A,2),:)

@time begin
  prev_rank = size(A,1)
  cur_size = size(A,2)

  L,p = RBSteady._cholesky_decomp(X[2])
  L′ = kron(I(prev_rank),L)
  p′ = vec(((collect(1:prev_rank).-1)*cur_size .+ p')')

  M = reshape(A,prev_rank*cur_size,:)
  Ur,Sr,Vr = RBSteady.tpod(red_style[1],M,L′,p′)

  core = reshape(Ur,prev_rank,cur_size,:)
  remainder = Sr.*Vr'
end

function _ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3},X::AbstractSparseMatrix) where T
  prev_rank = size(A,1)
  cur_size = size(A,2)

  L,p = RBSteady._cholesky_decomp(X)

  XA = _tt_mul(L,p,A)
  XM = reshape(XA,:,size(XA,3))

  Ũr,Sr,Vr = RBSteady.standard_tpod(red_style,XM)
  c̃ = reshape(Ũr,prev_rank,cur_size,:)
  core = _tt_div(L,p,c̃)
  remainder = Sr.*Vr'

  return core,remainder
end

function _tt_mul(L::AbstractSparseMatrix{T},p::Vector{Int},A::AbstractArray{T,3}) where T
  @check size(L,1) == size(L,2) == size(A,2)
  B = similar(A)
  Ap = A[:,p,:]
  @inbounds for i1 in axes(B,1)
    B[i1,:,:] = L'*Ap[i1,:,:]
  end
  return B
end

function _tt_div(L::AbstractSparseMatrix{T},p::Vector{Int},A::AbstractArray{T,3}) where T
  @check size(L,1) == size(L,2) == size(A,2)
  B = similar(A)
  @inbounds for i1 in axes(B,1)
    B[i1,:,:] = L'\A[i1,:,:]
  end
  return B
end
