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
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=5,nsnaps_mdeim=20)
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

using LinearAlgebra
using SparseArrays

mat = fesnaps
N = 4
N_space = 2
T = Float64
cores1 = Vector{Array{T,3}}(undef,N-1)
ranks1 = fill(1,N)
sizes = size(mat)
M1 = RBSteady.ttsvd!((cores1,ranks1,sizes),mat,X[1][1];ids_range=1)
M1 = RBSteady.ttsvd!((cores1,ranks1,sizes),M1,kron(X[1][2],I(ranks1[2]));ids_range=2)

cores2 = Vector{Array{T,3}}(undef,N-1)
ranks2 = fill(1,N)
M2 = RBSteady.ttsvd!((cores2,ranks2,sizes),mat,X[2][1];ids_range=1)
M2 = RBSteady.ttsvd!((cores2,ranks2,sizes),M2,kron(X[2][2],I(ranks2[2]));ids_range=2)

M = reshape(vcat(reshape(M1,:,10),reshape(M2,:,10)),:,10,10)
ranks = [1,ranks1[2]+ranks2[2],ranks1[3]+ranks2[3],1]
M = RBSteady.ttsvd!((cores1,ranks,sizes),M;ids_range=3)

function _weight_array!(weights,cores,X,::Val{1})
  X1 = getindex.(get_decomposition(X),1)
  K = length(X1)
  core = cores[1]
  rank = size(core,3)
  W = zeros(rank,K,rank)
  w = zeros(size(core,2))
  @inbounds for k = 1:K
    X1k = X1[k]
    for i′ = 1:rank
      mul!(w,X1k,core[1,:,i′])
      for i = 1:rank
        W[i,k,i′] = core[1,:,i]'*w
      end
    end
  end
  weights[1] = W
  return
end

function _weight_array!(weights,cores,X,::Val{d}) where d
  X1 = getindex.(get_decomposition(X),d)
  K = length(Xd)
  W_prev = weights[d-1]
  core = cores[d]
  rank = size(core,3)
  rank_prev = size(W_prev,3)
  W = zeros(rank,K,rank)
  w = zeros(size(core,2))
  @inbounds for k = 1:K
    Xdk = Xd[k]
    @views Wk = W[:,k,:]
    @views Wk_prev = W_prev[:,k,:]
    for i′_prev = 1:rank_prev
      for i′ = 1:rank
        mul!(w,Xdk,core[i′_prev,:,i′])
        for i_prev = 1:rank_prev
          Wk_prev′ = Wk_prev[i_prev,i′_prev]
          for i = 1:rank
            Wk[i,i′] += Wk_prev′*core[i_prev,:,i]'*w
          end
        end
      end
    end
  end
  weights[d] = W
  return
end

c1 = BlockCore([cores1[1],cores2[1]],[true,true])
c2 = BlockCore([cores1[2],cores2[2]],[true false
  false true])
c3 = cores1[end]

# weights = Vector{Array{T,3}}(undef,1)
c1′,r1 = RBSteady.reduce_rank(c1)
c2′ = RBSteady.absorb(c2,r1)
# _weight_array!(weights,[c1′,c2′],X,Val{1}())
# XW = RBSteady._get_norm_matrix_from_weight(X,weights[end])
# c2′,r2 = RBSteady.reduce_rank(c2′,XW)
c2′,r2 = RBSteady.reduce_rank(c2′)
c3′,_ = RBSteady.reduce_rank(RBSteady.absorb(c3,r2))

Φs′ = cores2basis(c1′,c2′)
Φs′'*kron(X)*Φs′
Φst′ = cores2basis(c1′,c2′,c3′)

s = flatten_snapshots(fesnaps)
s - Φs′*Φs′'*kron(X)*s
sst - Φst′*Φst′'*Xst*sst
sst - Φst′*Φst′'*sst

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
using PartitionedArrays
cores_k,ranks_k,mats = map(1:rank(X)) do k
  cores_k = copy(cores[ids_range])
  ranks_k = copy(ranks)
  mat_k = RBSteady.ttsvd!((cores_k,ranks_k,sizes),mat,X[k];ids_range)
  cores_k,ranks_k,mat_k
end |> tuple_of_arrays
for d in ids_range
  touched = d == first(ids_range) ? fill(true,rank(X)) : I(rank(X))
  cores_d = getindex.(cores_k,d)
  cores[d] = BlockCore(cores_d,touched)
end
R = RBSteady.orthogonalize!(cores[ids_range],X)

M = vcat(map(m->reshape(m,:,size(m,3)),mats)...)
absorb(mat,R)
# weight = ones(1,rank(X),1)
# d,core = 2,cores[2]
# decomp = get_decomposition(X)
# if d == length(cores)
#   XW = RBSteady._get_norm_matrix_from_weight(X,weight)
#   core′,R = RBSteady.reduce_rank(core,XW)
#   cores[d] = core′
#   return R
# end
# next_core = cores[d+1]
# Xd = getindex.(decomp,1)
# core′,R = RBSteady.reduce_rank(core)
# cores[d] = core′
# cores[d+1] = RBSteady.absorb(next_core,R)
# weight = RBSteady._weight_array(weight,core′,Xd)

cores = ttsvd(fesnaps,X)

Xglobal = kron(I(10),kron(X))
basis = cores2basis(cores...)
basis'*Xglobal*basis

S = copy(fesnaps)
SST = reshape(S,:,10)
ee = SST - basis*basis'*Xst*SST

bbasis = reduced_basis(fesnaps,X)
# more tests
function _orthogonalize!(cores,X::AbstractTProductTensor)
  weight = ones(1,rank(X),1)
  decomp = get_decomposition(X)
  for d in eachindex(cores)
    core = cores[d]
    if d == length(cores)
      core′,R = RBSteady.reduce_rank(core)
      cores[d] = core′
      return
    end
    next_core = cores[d+1]
    Xd = getindex.(decomp,d)
    if d == rank(X)
      XW = RBSteady._get_norm_matrix_from_weight(X,weight)
      core′,R = RBSteady.reduce_rank(core,XW)
    else
      core′,R = RBSteady.reduce_rank(core)
      weight = RBSteady._weight_array(weight,core′,Xd)
    end
    cores[d] = core′
    # ranks[d+1] = size(core′,3)
    cores[d+1] = RBSteady.absorb(next_core,R)
  end
end

function _orthogonalize!(cores)
  for d in eachindex(cores)
    core = cores[d]
    if d == length(cores)
      core′,R = RBSteady.reduce_rank(core)
      cores[d] = core′
      return
    end
    next_core = cores[d+1]
    core′,R = RBSteady.reduce_rank(core)
    cores[d] = core′
    cores[d+1] = RBSteady.absorb(next_core,R)
  end
end

mat = fesnaps
cores1 = ttsvd(mat,X[1])
cores2 = ttsvd(mat,X[2])

core_block_1 = BlockCore([cores1[1],cores2[1]],[true,true])
core_block_2 = BlockCore([cores1[2],cores2[2]],I(2))
core_block_3 = BlockCore([cores1[3],cores2[3]],I(2))

allcores = Array{Float64}.([core_block_1,core_block_2,core_block_3])
_orthogonalize!(allcores,X)

Xglobal = kron(I(10),kron(X))
basis = cores2basis(allcores...)
basis'*Xglobal*basis

ee = SST - basis*basis'*Xglobal*SST

allcores = Array{Float64}.([core_block_1,core_block_2,core_block_3])
_orthogonalize!(allcores)

basis = cores2basis(allcores...)
basis'*basis
eest = SST - basis*basis'*SST

# basis = cores2basis(allcores[1:2]...)
# basis'*basis
# eest = s - basis*basis'*s

basis_space = truncated_pod(s)
compressed_s2 = compress(s,basis_space;swap_mode=true)
basis_time = truncated_pod(compressed_s2)
basis_spacetime = kron(basis_time,basis_space)

SST - basis_spacetime*basis_spacetime'*SST
