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
using SparseArrays
using LinearAlgebra
using Kronecker
using Mabla.FEM
using Mabla.RB

θ = 1.0
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (3,3)
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

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=20,nsnaps_test=1,nsnaps_mdeim=10)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

params = [
  [0.1,0.9,0.5],
  [0.2,0.4,0.8],
  [0.3,0.7,0.4],
  [0.9,0.2,0.4],
  [0.5,0.5,0.6],
  [0.8,0.4,0.2],
  [0.3,0.4,0.3],
  [0.1,0.2,0.9],
  [0.9,0.2,0.1],
  [0.4,0.6,0.5],
  [0.2,0.5,0.5],
  [0.1,0.2,1.0],
  [0.2,0.7,0.1],
  [0.2,0.2,0.2],
  [0.9,0.5,0.1],
  [0.8,0.7,0.2],
  [0.1,0.1,0.7],
  [0.1,0.7,0.9],
  [0.4,0.4,0.1],
  [0.4,0.3,0.5],
  [0.2,0.3,0.6]
]
r = TransientParamRealization(ParamRealization(params),t0:dt:tf)

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ;r)

# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
# results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

# println(RB.space_time_error(results))

# test 1 : try old code with new basis
soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
red_trial,red_test = reduced_fe_space(rbsolver,feop,soff)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
jjac,rres = jacobian_and_residual(rbsolver,pop,smdeim)
red_res = RB.reduced_residual(rbsolver,pop,rres)
# U,V = trial(nothing),test
# sparsity = get_sparsity(U,V)
# A = jjac[1][1]
# mdeim_style = rbsolver.mdeim_style
# basis = reduced_basis(A;ϵ=RB.get_tol(rbsolver))
# temp_basis = RB.temp_reduced_basis(A,sparsity;ϵ=RB.get_tol(rbsolver))
# lu_interp,integration_domain = RB.temp_mdeim(mdeim_style,temp_basis...)
red_jac = RB.temp_reduced_jacobian(rbsolver,pop,jjac)
trians_rhs = get_domains(red_res)
trians_lhs = map(get_domains,red_jac)
new_op = change_triangulation(pop,trians_rhs,trians_lhs)
rbop = PODMDEIMOperator(new_op,red_jac,red_res)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)
println(RB.space_time_error(results))

# test 2: test goodness of fit of basis
mat = jjac[1][1]
T,N = Float64,4
cores = Vector{Array{T,3}}(undef,N)
ranks = fill(1,N)
sizes = size(mat)
cache = cores,ranks,sizes
M = RB.ttsvd!(cache,copy(mat);ids_range=1:N-1)
cores[end] = M

c2m = cores2basis(cores...)
ss = reshape(c2m,size(mat))
e = mat - ss

N1 = 3
mat1 = reshape(mat,:,10,10)
cores1 = Vector{Array{T,3}}(undef,N1)
ranks1 = fill(1,N1)
sizes1 = size(mat1)
cache1 = cores1,ranks1,sizes1
M1 = RB.ttsvd!(cache1,copy(mat1);ids_range=1:N1-1)
cores1[end] = M1

c2m1 = cores2basis(cores1...)
ss1 = reshape(c2m1,size(mat1))
e1 = mat1 - ss1

mat2 = jjac[1][1]
T2,N2 = Float64,4
cores2 = Vector{Array{T,3}}(undef,N2)
ranks2 = fill(1,N2)
sizes2 = size(mat2)
cache2 = cores2,ranks2,sizes2
tol = [1e-8,1e-4,1e-4]
M2 = RB.temp_ttsvd!(cache2,copy(mat2),tol;ids_range=1:N2-1)
cores2[end] = M2

c2m2 = cores2basis(cores2...)
ss2 = reshape(c2m2,size(mat2))
e2 = mat2 - ss2

function old_ttsvd(X;kwargs...)
  d = ndims(X)
  n = size(X)

  ranks = fill(1,d)
  cores = Vector{Array{eltype(X),3}}(undef, ndims(X))
  T = X

  for k = 1:d-1
    if k == 1
      X_k = reshape(T, n[k], :)
    else
      X_k = reshape(T, ranks[k-1] * n[k], :)
    end

    U,S,V = svd(X_k)
    r = RB.truncation(S;kwargs...)
    U = U[:, 1:r]
    S = S[1:r]
    V = V[:, 1:r]

    ranks[k] = r
    T = reshape(S.*V', ranks[k], n[k+1], :)
    if k == 1
      cores[k] = reshape(U, 1, n[k], ranks[k])
    else
      cores[k] = reshape(U, ranks[k-1], n[k], ranks[k])
    end
  end
  cores[d] = reshape(T, ranks[d-1], n[d], 1)
  return cores
end

function my_eval(cores)
  function fun(cores,i)
    output = 1
    for k = eachindex(cores)
      C = cores[k]
      output = output * C[:, i[k], :]
    end
    return output[1]
  end

  sizes = Tuple(map(x->size(x,2),cores))
  OUT = zeros(sizes)
  for i in CartesianIndices(sizes)
    OUT[i] = fun(cores,Tuple(i))
  end
  return OUT
end

function Base.:*(x::Vector{Float64}, y::Vector{Float64})
  z = kronecker(y, x)
  return z[:]
end

b = old_ttsvd(mat)
matrec = my_eval(b)

# index map
U,V = test,test
sparsity = get_sparsity(U,V)
psparsity = FEM.permute_sparsity(sparsity,U,V)
I,J,_ = findnz(psparsity)
i,j,_ = FEM.univariate_findnz(psparsity)

IJ = get_nonzero_indices(psparsity)
lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

unrows = FEM.univariate_num_rows(sparsity)
uncols = FEM.univariate_num_cols(sparsity)
unnz = FEM.univariate_nnz(sparsity)
g2l = zeros(Int,unnz...)

k,gid = 2,IJ[2]
irows = Tuple(tensorize_indices(I[k],unrows))
icols = Tuple(tensorize_indices(J[k],uncols))
iaxes = CartesianIndex.(irows,icols)
global2local = map((i,j) -> findfirst(i.==[j]),lids,iaxes)
g2l[global2local...] = gid



index_map_I = get_dof_permutation(V)
index_map_J = get_dof_permutation(U)
index_map_I_univ1 = get_dof_permutation(Float64,model.models_1d[1],V.spaces_1d[1],order)
index_map_I_univ2 = get_dof_permutation(Float64,model.models_1d[2],V.spaces_1d[2],order)

# Iperm = map(i->findfirst(index_map_I[:].==i),I)
# Jperm = map(j->findfirst(index_map_J[:].==j),J)
# iperm = [
#   map(i->findfirst(index_map_I_univ1[:].==i),i[1]),
#   map(i->findfirst(index_map_I_univ2[:].==i),i[2])
#   ]
# jperm = [
#   map(j->findfirst(index_map_I_univ1[:].==j),j[1]),
#   map(j->findfirst(index_map_I_univ2[:].==j),j[2])
#   ]
Iperm = map(i->findfirst(I.==i),index_map_I[:])
Jperm = map(j->findfirst(J.==j),index_map_J[:])
iperm = [
  map(i->findfirst(i[1].==i),index_map_I_univ1[:]),
  map(i->findfirst(i[2].==i),index_map_I_univ2[:])
  ]
jperm = [
  map(j->findfirst(j[1].==j),index_map_I_univ1[:]),
  map(j->findfirst(j[2].==j),index_map_I_univ2[:])
  ]

IJ = get_nonzero_indices(sparsity)
lids = map((ii,ji)->CartesianIndex.(ii,ji),iperm,jperm)

unrows = FEM.univariate_num_rows(sparsity)
uncols = FEM.univariate_num_cols(sparsity)
unnz = FEM.univariate_nnz(sparsity)
g2l = zeros(Int,unnz...)

@inbounds for (k,gid) = enumerate(IJ)
  irows = Tuple(tensorize_indices(Iperm[k],unrows))
  icols = Tuple(tensorize_indices(Jperm[k],uncols))
  iaxes = CartesianIndex.(irows,icols)
  global2local = map((i,j) -> findfirst(i.==[j]),lids,iaxes)
  g2l[global2local...] = gid
end

MS = M[index_map_I[:],index_map_I[:]]
MSB1 = MS[1:6,1:6]
v1 = nonzeros(MSB1)

M1 = assemble_matrix((u,v)->∫(u*v)dΩ.measures_1d[1],test.spaces_1d[1],test.spaces_1d[1])
M1p = M1[index_map_I_univ1,index_map_I_univ1]
M2 = assemble_matrix((u,v)->∫(u*v)dΩ.measures_1d[2],test.spaces_1d[2],test.spaces_1d[2])
M2p = M2[index_map_I_univ2,index_map_I_univ2]
M12 = kron(M2,M1)
M12p = kron(M2p,M1p)

p = test.dof_permutation
ip = invperm(p[:])
