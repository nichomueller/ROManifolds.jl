using Gridap
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

# time marching
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

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
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

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
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)
ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=2)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(compute_error(results))
println(compute_speedup(results))

save(test_dir,fesnaps)
save(test_dir,rbop)

X = assemble_norm_matrix(feop)
# X1 = X.arrays_1d[1]
# X2 = X.arrays_1d[2]
# Xk = kron(X2,X1)
Xk = kron(X)
H = cholesky(Xk)
L,p = sparse(H.L),H.p

using LinearAlgebra
using SparseArrays

# # case 1
# mat = fesnaps
# N = 4
# N_space = 2
# T = Float64
# cores = Vector{Array{T,3}}(undef,N-1)
# ranks = fill(1,N)
# sizes = size(mat)
# M = RBSteady.ttsvd!((cores,ranks,sizes),mat,X1;ids_range=1)
# M = RBSteady.ttsvd!((cores,ranks,sizes),M,kron(X2,I(ranks[2]));ids_range=2)
# _ = RBSteady.ttsvd!((cores,ranks,sizes),M;ids_range=3)

# Φ = cores2basis(cores[1:2]...)
# s = flatten_snapshots(fesnaps)
# _e = s - Φ*Φ'*Xk*s

# # Φ = cores2basis(cores...)
# # Mat = reshape(fesnaps,:,10)
# # Xk_st = kron(I(10),Xk)
# # _e = Mat - Φ*Φ'*Xk_st*Mat

# # case 2
# s′ = L'*s[p,:]
# t′ = permutedims(reshape(s′,size(fesnaps)),(1,2,4,3))
# cores′ = ttsvd(t′)
#   mat′ = t′
#   c′ = Vector{Array{T,3}}(undef,N-1)
#   r′ = fill(1,N)
#   M′ = RBSteady.ttsvd!((c′,r′,sizes),mat′;ids_range=1)
#   M′ = RBSteady.ttsvd!((c′,r′,sizes),M′;ids_range=2)
#   M′ = RBSteady.ttsvd!((c′,r′,sizes),M′;ids_range=3)

# # # cores′[1] = L'\cores′[1][invperm(p),:]
# H1 = cholesky(X1)
# L1,p1 = sparse(H1.L),H1.p
# c1′ = (L1'\reshape(cores′[1],:,size(cores′[1],3)))[invperm(p1),:]
# c1′ = reshape(c1′,1,size(c1′)...)

# # H2 = cholesky(kron(X2,I(9)))
# # L1,p1 = sparse(H1.L),H1.p
# # c1′ = L1'\reshape(cores′[1],:,size(cores′[1],3))[invperm(p1),:]
# # c1′ = reshape(c1′,1,size(c1′)...)

# Φ′ = (L'\cores2basis(cores′[1:2]...))[invperm(p),:]
# _e′ = s - Φ′*Φ′'*Xk*s

# gradient case
mat = fesnaps
N = 4
N_space = 2
T = Float64
cores1 = Vector{Array{T,3}}(undef,N-1)
ranks1 = fill(1,N)
sizes = size(mat)
M1 = RBSteady.ttsvd!((cores1,ranks1,sizes),mat,X.gradients_1d[1];ids_range=1)
M1 = RBSteady.ttsvd!((cores1,ranks1,sizes),M1,kron(X.arrays_1d[2],I(ranks1[2]));ids_range=2)

cores2 = Vector{Array{T,3}}(undef,N-1)
ranks2 = fill(1,N)
M2 = RBSteady.ttsvd!((cores2,ranks2,sizes),mat,X.arrays_1d[1];ids_range=1)
M2 = RBSteady.ttsvd!((cores2,ranks2,sizes),M2,kron(X.gradients_1d[2],I(ranks2[2]));ids_range=2)

M = reshape(vcat(reshape(M1,:,10),reshape(M2,:,10)),:,10,10)
ranks = [1,ranks1[2]+ranks2[2],ranks1[3]+ranks2[3],1]
M = RBSteady.ttsvd!((cores1,ranks,sizes),M;ids_range=3)

using BlockArrays

# struct BlockCore{T,N,D,A<:AbstractArray{T,D},BS} <: AbstractArray{T,D}
#   array::Vector{A}
#   touched::Array{Bool,N}
#   axes::BS
#   function BlockCore(array::Vector{A},touched::Array{Bool,N},axes::BS) where {T,N,D,A<:AbstractArray{T,D},BS}
#     @assert all((size(a,2)==size(first(array),2) for a in array))
#     new{T,N,D,A,BS}(array,touched,axes)
#   end
# end

# function BlockCore(array::Vector{<:AbstractArray},touched::AbstractArray{Bool})
#   block_sizes = _sizes_from_blocks(array,touched)
#   axes = map(blockedrange,block_sizes)
#   BlockCore(array,touched,axes)
# end

# const _BlockVectorCore{T,D,A<:AbstractArray{T,D}} = BlockCore{T,1,D,A}
# const _BlockMatrixCore{T,D,A<:AbstractArray{T,D}} = BlockCore{T,2,D,A}
# const _BlockVectorCore3D{T} = _BlockVectorCore{T,3,Array{T,3}}
# const _BlockMatrixCore3D{T} = _BlockMatrixCore{T,3,Array{T,3}}

# # Base.size(a::_BlockVectorCore3D) = (1,size(a.array[1],2),length(axes(a,3)))
# # Base.size(a::_BlockMatrixCore3D) = (length(axes(a,1)),size(a.array[1],2),length(axes(a,3)))
# Base.size(a::BlockCore) = map(length,axes(a))
# Base.axes(a::BlockCore) = a.axes

# function Base.getindex(a::_BlockVectorCore3D{T},i::Vararg{Integer,3}) where T
#   i1,i2,i3 = i
#   @assert i1 == 1
#   b3 = BlockArrays.findblockindex(axes(a,3),i3)
#   a.array[b3.I...][i1,i2,b3.α...]
# end

# function Base.getindex(a::_BlockMatrixCore3D{T},i::Vararg{Integer,3}) where T
#   i1,i2,i3 = i
#   b1 = BlockArrays.findblockindex(axes(a,1),i1)
#   b3 = BlockArrays.findblockindex(axes(a,3),i3)
#   if b1.I == b3.I
#     a.array[b1.I...][b1.α...,i2,b3.α...]
#   else
#     zero(T)
#   end
# end

# function _sizes_from_blocks(a::Vector{<:AbstractArray},touched::Vector{Bool})
#   s1 = fill(1,length(a))
#   s2 = fill(size(a[1],2),length(a))
#   s3 = map(a -> size(a,3),a)
#   for i in 1:length(a)-1
#     s1[i] = 0
#     s2[i] = 0
#   end
#   return (s1,s2,s3)
# end

# function _sizes_from_blocks(a::Vector{<:AbstractArray},touched::Matrix{Bool})
#   s1 = map(a -> size(a,1),a)
#   s2 = fill(size(a[1],2),length(a))
#   s3 = map(a -> size(a,3),a)
#   for i in 1:length(a)-1
#     s2[i] = 0
#   end
#   return (s1,s2,s3)
# end

c1 = BlockCore([cores1[1],cores2[1]],[true,true])
c2 = BlockCore([cores1[2],cores2[2]],[true false
  false true])
c3 = cores1[end]

Φ = cores2basis(c1,c2,c3)

Φs = cores2basis(c1,c2)
Φs'*Xk*Φs

weights = Vector{Array{T,3}}(undef,1)
q1,r1 = RBSteady.pivoted_qr(c1[1,:,:])
c1′ = reshape(q1,1,size(c1,2),:)
c2′,r2 = RBSteady.absorb(c2,r1)
RBSteady._weight_array!(weights,[c1′,c2′],X,Val{1}())
_ = RBSteady.orthogonalize!(c2′,X,weights)
c3′,_ = RBSteady.absorb(c3,r2)

Φs′ = cores2basis(c1′,c2′)
Φs′'*Xk*Φs′

s - Φs′*Φs′'*Xk*s

# RBSteady.absorb(c2′,r1)
AA,BB,CC = size(c2′)
AA′ = size(r1,1)
Rcore = r1*reshape(c2′,size(c2′,1),:)
@assert size(Rcore) == (AA′,BB*CC)
inp = reshape(Rcore,:,size(c2′,3))
Q̃,_ = RBSteady.pivoted_qr(reshape(Rcore,:,size(c2′,3)))
@assert size(Q̃) == (AA′*BB,CC) # NOT TRUE
c2′ = reshape(Q̃,AA′,BB,:)


# XW = RBSteady._get_norm_matrix_from_weights(X,weights)
# with ttsvd

using LinearAlgebra
mat = fesnaps
N,T = 4,Float64
N_space = N-2
cores = Vector{Array{T,3}}(undef,N-1)
ranks = fill(1,N)
sizes = size(mat)
K = rank(X)
ids_range = 1:N_space
cores′ = cores[ids_range]
ranks′ = ranks
k_cores = fill(cores′,K)
k_ranks = fill(ranks′,K)
mats = ()
for k in 1:K
  ck = k_cores[k]
  rk = k_ranks[k]
  Xk = X[k]
  mat_k = RBSteady.ttsvd!((ck,rk,sizes),mat,Xk;ids_range)
  # d = ids_range[2]
  # mat_d = reshape(M,ranks[d]*sizes[d],:)
  # Ur,Σr,Vr = RBSteady._tpod(mat_d,X[d])
  # r = size(Ur,2)
  # ranks[d+1] = r
  # M = reshape(Σr.*Vr',r,sizes[d+1],:)
  # ck[d] = reshape(Ur,ranks[d],sizes[d],r)
  mats = (mats...,mat_k)
end
bcores = BlockCore(k_cores)
R = RBSteady.orthogonalize!(bcores,X)
mat = vec(mats...)
return absorb(mat,R)
