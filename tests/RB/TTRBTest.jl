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
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=2)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
# norm_matrix = assemble_norm_matrix(feop)
# cores = ttsvd(soff,norm_matrix)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
A,b = jacobian_and_residual(rbsolver,pop,smdeim)

# b1,t1 = b.values[1],b.trians[1]
# bred = RB.reduced_form(rbsolver,b1,t1,get_test(pop))
# mdeim_style = rbsolver.mdeim_style
# basis = reduced_basis(b1;ϵ=RB.get_tol(rbsolver))
# lu_interp,integration_domain = mdeim(mdeim_style,basis)
# proj_basis = reduce_operator(mdeim_style,basis,get_test(pop))

A1,t1 = A[1].values[1],A[1].trians[1]
# Ared = RB.reduced_form(rbsolver,A1,t1,get_trial(pop),get_test(pop))
mdeim_style = rbsolver.mdeim_style
basis = reduced_basis(A1;ϵ=RB.get_tol(rbsolver))
lu_interp,integration_domain = mdeim(mdeim_style,basis)
proj_basis = reduce_operator(mdeim_style,basis,args...)

# bs = get_basis_space(basis)
bs = basis.cores[1:end-1]
STOP
# fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
# results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

# println(RB.space_time_error(results))
# println(RB.speedup(results))

# save(test_dir,fesnaps)
# save(test_dir,rbop)
# save(test_dir,results)

# form(u,v) = ∫(u*v)dΩ
# M = assemble_matrix(form,trial(nothing),test)
# p = FEM.get_free_dof_permutation(test)
# _Mp = M[p,p]
# Mp = permutedims(M[p,p],[1,3,2,4])

# boh = rand(4,5,4,5)
# permutedims(boh,[1,3,2,4])

# using SparseArrays
# I,J,V = findnz(M)

# using PartitionedArrays
# Ix,Iy = tuple_of_arrays(map(i->Tuple(findfirst(p .== i)),I))
# Jx,Jy = tuple_of_arrays(map(i->Tuple(findfirst(p .== i)),J))
# Ixy = map(i->findfirst(p .== i),I)
# Jxy = map(i->findfirst(p .== i),J)
# psparseI = p[Ixy]
# # ci = map(CartesianIndex,I,J)

# ids = CartesianIndices(size(p))

# M = assemble_norm_matrix(form,trial(nothing),test)
# Mk = kron(M.arrays_1d[1],M.arrays_1d[2])


function vector_prod(a::TTSVDCores,basis_test::TTSVDCores)
  cores_hat = map(compress_core,a.cores,basis_test.cores)
  multiply_cores(cores_hat...)
end

function matrix_prod(a::TTSVDCores,basis_trial::TTSVDCores,basis_test::TTSVDCores)
  cores_hat = map(compress_core,a.cores,basis.cores)
  multiply_cores(cores_hat...)
end

function compress_core(a::Array{T,3},b::Array{S,3}) where {T,S}
  TS = promote_type(T,S)
  ab = zeros(TS,size(a,1),size(b,1),size(a,3),size(b,3))
  @inbounds for i = CartesianIndices(size(ab))
    ia1,ib1,ia3,ib3 = Tuple(i)
    ab[i] = b[ib1,:,ib3]'*a[ia1,:,ia3]
  end
  return ab
end

function _multiply_cores(c1::Array,cores::Array...)
  @check size(first(c1),1) == size(first(c1),2) == 1
  _c1,_cores... = cores
  _multiply_cores(_multiply_cores(c1,_c1),_cores...)
end

function _multiply_cores(a::Array{T,4},b::Array{S,4}) where {T,S}
  @check (size(a,3)==size(b,1) && size(a,4)==size(b,2))
  TS = promote_type(T,S)
  ab = zeros(TS,size(a,1),size(a,2),size(b,3),size(b,4))
  @inbounds for i = CartesianIndices(size(ab))
    ia1,ia2,ib3,ib4 = Tuple(i)
    ab[i] = dot(a[ia1,ia2,:,:],b[:,:,ib3,ib4])
  end
  return ab
end

function multiply_cores(cores::Array...)
  core = _multiply_cores(cores...)
  dropdims(core;dims=(1,2))
end

bs = RB.get_basis(red_test)
b1,t1 = b.values[1],b.trians[1]
bred1 = reduced_basis(b1;ϵ=RB.get_tol(rbsolver))
proj_basis = reduce_operator(rbsolver.mdeim_style,bred1,RB.get_basis(red_test))
_proj_basis = vector_prod(bred1,bs)

# modify ttsvd for matrices (4-D cores instead of 3-D)
struct MatrixTTSVDCores{A,B} <: Projection
  space_cores::A
  time_core::B
end

function mat_ttsvd!(cache,mat::AbstractArray{T,N},args...;ids_range=1:Int(N/2-1),kwargs...) where {T,N}
  cores,ranks,sizes = cache
  sizes_space = eachcol(reshape(collect(sizes),2,:))[1:end-1]
  for k in ids_range
    nrows_k,ncols_k = sizes_space[k]
    mat_k = reshape(mat,ranks[k]*nrows_k*ncols_k,:)
    U,Σ,V = svd(mat_k)
    rank = RB.truncation(Σ;kwargs...)
    core_k = U[:,1:rank]
    ranks[k+1] = rank
    mat = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[2*k+1]...,:)
    cores[k] = reshape(core_k,ranks[k],nrows_k,ncols_k,rank)
  end
  return mat
end

function mat_ttsvd(mat::AbstractArray{T,N},X=nothing;kwargs...) where {T,N}
  M = Int(N/2)
  space_cores = Vector{Array{T,4}}(undef,M-1)
  time_core = Vector{Array{T,3}}(undef,1)
  ranks = fill(1,M)
  sizes = size(mat)
  # routine on the spatial indexes
  M = mat_ttsvd!((space_cores,ranks,sizes),copy(mat),X;ids_range=1:M-1,kwargs...)
  # routine on the temporal index
  _ = RB.ttsvd!((time_core,ranks,sizes),M;ids_range=N-1,kwargs...)
  return cores
end

boh = mat_ttsvd(A[1][1])

N = 6
M = Int(N/2)
mat = A[1][1]
space_cores = Vector{Array{Float64,4}}(undef,M-1)
ranks = fill(1,M)
sizes = size(mat)
M = mat_ttsvd!((space_cores,ranks,sizes),copy(mat);ids_range=1:M-1)
time_core = Vector{Array{Float64,3}}(undef,1)
_ = RB.ttsvd!((time_core,ranks,sizes),M;ids_range=N-1)
