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

n = 30
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
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","tensor_train")))

# fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
# results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)
fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,rbstats,rbstats)

compute_error(results)

# comparison with standard TT-SVD
sTT = fesnaps
X = assemble_norm_matrix(feop)
@time ttsvd(sTT[1],X[1])
basis = Projection(sTT,X)
b1 = basis[1]
cores1 = get_cores(b1)
RBTransient.check_orthogonality(cores1,X[1])

#supr
B = assemble_coupling_matrix(feop)
basis′ = enrich_basis(basis,X,B)
b1′ = basis′[1]
cores1′ = get_cores(b1′)
basis_space = dropdims(RBSteady._cores2basis(cores1′[1:2]...);dims=1)
basis_space'*Xglobal_space*basis_space

# c1 = cores1′[1]
# cb1,cb2 = blocks(c1)
# cb1,cb2 = cb1[1,:,:],cb2[1,:,:]
# gram_schmidt!(cb2,cb1)
# newcore1 = reshape(hcat(cb1,cb2),1,size(hcat(cb1,cb2))...)
# weights = Vector{Array{Float64,3}}(undef,D-1)
# RBSteady._weight_array!(weights,[newcore1],norms_primal[1],Val{1}())
# newcores =

cores_space = get_cores_space(basis)
core_time = RBTransient.get_core_time(basis)
pblocks = (1,2)
dblocks = (3,)
cores_primal_space = map(ip -> cores_space[ip],pblocks)
core_primal_time = map(ip -> core_time[ip],pblocks)
cores_dual_space = map(id -> cores_space[id],dblocks)
core_dual_time = map(id -> core_time[id],dblocks)
norms_primal = map(ip -> X[Block(ip,ip)],pblocks)

id = 3
rcores_space = Vector{Array{Float64,3}}[]
rcore_time = Array{Float64,3}[]
rcore = Matrix{Float64}[]
cores_dual_space_i = cores_space[id]
core_dual_time_i = core_time[id]
for ip in eachindex(pblocks)
  A = X[Block(ip,ip)]
  C = B[Block(ip,id)]
  cores_primal_space_i = cores_space[ip]
  core_primal_time_i = core_time[ip]
  RBSteady.reduced_coupling!((rcores_space,rcore_time,rcore),cores_primal_space_i,core_primal_time_i,
    cores_dual_space_i,core_dual_time_i,A,C)
end
rcore = vcat(rcore...)

i = 1
D = length(cores_primal_space[1])
weights = Vector{Array{Float64,3}}(undef,D-1)
cores_primal_space[1][1] = cat_cores(cores_primal_space[1][1],rcores_space[1][1])
RBSteady._weight_array!(weights,cores_primal_space[1],norms_primal[1],Val{1}())

cores_primal_space[1][2] = cat_cores(cores_primal_space[1][2],rcores_space[1][2])
R = RBSteady.orthogonalize!(cores_primal_space[1][2],norms_primal[1],weights)
# core_primal_time = RBSteady.pushlast(core_primal_time[1],rcore_time[1][:,:,i])
aa = core_primal_time[1]
bb = rcore_time[1][:,:,i]
s1,s2,s3 = size(aa)
s1′ = size(bb,1)
boh = zeros(aa,s1,s2,s3+1)
@views boh[:,:,1:s3] = aa
@views boh[s1-s1′+1:end,:,s3+1] = bb

RBSteady.absorb!(core_primal_time,R)

R = nothing
for ip in eachindex(cores_primal_space)
  R = RBSteady.add_and_orthogonalize(cores_primal_space′[ip],core_primal_time′[ip],
    rcores_space[ip],rcore_time[ip],norms_primal[ip],R,i;flag=false)
end
rcore = RBSteady._update_reduced_coupling(cores_primal_space′,core_primal_time′,rcores_space,rcore_time,rcore)

# TPOD
using Mabla.FEM.IndexMaps
using BlockArrays
using SparseArrays
using LinearAlgebra
_dΩ = dΩ.measure

_coupling((u1,u2,p),(v1,v2,q)) = ∫(p*∂ₓ₁(v1))_dΩ + ∫(p*∂ₓ₂(v2))_dΩ
_induced_norm((u1,u2,p),(v1,v2,q)) = ∫(v1*u1)_dΩ + ∫(v2*u2)_dΩ + ∫(∇(v1)⊙∇(u1))_dΩ + ∫(∇(v2)⊙∇(u2))_dΩ + ∫(p*q)_dΩ

_test_u = TestFESpace(model.model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
_trial_u1 = TransientTrialParamFESpace(_test_u,gμt_in_1)
_trial_u2 = TransientTrialParamFESpace(_test_u,gμt_in_2)
_test_p = TestFESpace(model.model,reffe_p;conformity=:C0)
_trial_p = TrialFESpace(_test_p)
_test = TransientMultiFieldParamFESpace([_test_u,_test_u,test_p];style=BlockMultiFieldStyle())
_trial = TransientMultiFieldParamFESpace([_trial_u1,_trial_u2,_trial_p];style=BlockMultiFieldStyle())
_feop = TransientParamLinearFEOperator((stiffness,mass),res,_induced_norm,ptspace,
  _trial,_test,_coupling,trian_res,trian_stiffness,trian_mass)

# basis (comp 1 velocity)
_s = change_index_map(TrivialIndexMap,fesnaps)
_X = assemble_norm_matrix(_feop)
@time Projection(_s[1],_X[Block(1,1)])
_basis = Projection(_s,_X)

# supr

#######

function test_weight_array1(weights,cores,X,::Val{1})
  X1 = tp_getindex(X,1)
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

weights1 = Vector{Array{Float64,3}}(undef,D-1)
test_weight_array1(weights1,cores_primal_space′[1],norms_primal[1],Val{1}())

function test_weight_array2(weights,cores,X,::Val{1})
  X1 = tp_getindex(X,1)
  K = length(X1)
  bcore = cores[1]
  offset = bcore.offset3
  rank = size(bcore,3)
  W = zeros(rank,K,rank)
  w = zeros(size(bcore,2))
  for (b,core) in enumerate(blocks(bcore))
    range = offset[b]+1:offset[b+1]
    @inbounds for k = 1:K
      X1k = X1[k]
      for (ib′,i′) = enumerate(range)
        mul!(w,X1k,core[1,:,ib′])
        for (ib,i) = enumerate(range)
          W[i,k,i′] = core[1,:,ib]'*w
        end
      end
    end
  end
  weights[1] = W
  return
end

weights2 = Vector{Array{Float64,3}}(undef,D-1)
test_weight_array2(weights2,cores_primal_space′[1],norms_primal[1],Val{1}())
