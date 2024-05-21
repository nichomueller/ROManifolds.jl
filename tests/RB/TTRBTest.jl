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
combine=(x,y)->θ*x+(1-θ)*y
mdeim_style = rbsolver.mdeim_style
basis = reduced_basis(A1;ϵ=RB.get_tol(rbsolver))
lu_interp,integration_domain = mdeim(mdeim_style,basis)
proj_basis = reduce_operator(mdeim_style,basis,get_trial(pop),get_test(pop);combine)

b1 = b[1]
basis_vec = reduced_basis(b1;ϵ=RB.get_tol(rbsolver))
proj_basis_vec = reduce_operator(mdeim_style,basis_vec,get_test(pop))
################### OLD INTERFACE #####################

struct OldBasicTTSnapshots{T,P,R} <: TTSnapshots{T,3}
  values::P
  realization::R
  function OldBasicTTSnapshots(values::P,realization::R) where {P<:ParamArray,R,D}
    T = eltype(P)
    new{T,P,R}(values,realization)
  end
end

function RB.BasicSnapshots(
  values::ParamArray,
  realization::TransientParamRealization)
  OldBasicTTSnapshots(values,realization)
end

function RB.tensor_getindex(s::OldBasicTTSnapshots,ispace,itime,iparam)
  s.values[iparam+(itime-1)*num_params(s)][ispace]
end

struct OldTransientTTSnapshots{T,P,R,V} <: TTSnapshots{T,3}
  values::V
  realization::R
  function OldTransientTTSnapshots(
    values::AbstractVector{P},
    realization::R,
    ) where {P<:ParamArray,R<:TransientParamRealization}

    V = typeof(values)
    T = eltype(P)
    new{T,P,R,V}(values,realization)
  end
end

function RB.TransientSnapshots(
  values::AbstractVector{P},
  realization::TransientParamRealization
  ) where P<:ParamArray
  OldTransientTTSnapshots(values,realization)
end

function RB.tensor_getindex(s::OldTransientTTSnapshots,ispace,itime,iparam)
  perm_ispace = s.permutation[ispace]
  s.values[itime][iparam][perm_ispace]
end

const BasicNnzTTSnapshots = OldBasicTTSnapshots{T,P,R} where {T,P<:FEM.ParamTTSparseMatrix,R}
const TransientNnzTTSnapshots = OldTransientTTSnapshots{T,P,R} where {T,P<:FEM.ParamTTSparseMatrix,R}
const NnzTTSnapshots = Union{BasicNnzTTSnapshots{T},TransientNnzTTSnapshots{T}} where {T}

RB.num_space_dofs(s::NnzTTSnapshots) = nnz(s.values[1])

function RB.tensor_getindex(s::BasicNnzTTSnapshots,ispace,itime,iparam)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace]
end

function RB.tensor_getindex(s::TransientNnzTTSnapshots,ispace,itime,iparam)
  nonzeros(s.values[itime][iparam])[ispace]
end

RB.sparsify_indices(s::BasicNnzTTSnapshots,srange::AbstractVector) = sparsify_indices(first(s.values),srange)
RB.sparsify_indices(s::TransientNnzTTSnapshots,srange::AbstractVector) = sparsify_indices(first(first(s.values)),srange)

function RB.select_snapshots_entries(s::NnzTTSnapshots,ispace,itime)
  RB._select_snapshots_entries(s,RB.sparsify_indices(s,ispace),itime)
end

function RB.get_nonzero_indices(s::NnzTTSnapshots)
  v = isa(s,OldBasicTTSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  return i .+ (j .- 1)*v.m
end

function RB.recast(s::NnzTTSnapshots,a::AbstractArray{T,3}) where T
  @check size(a,1) == 1
  v = isa(s,OldBasicTTSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  m,n = size(v)
  asparse = map(eachcol(dropdims(a;dims=1))) do v
    sparse(i,j,v,m,n)
  end
  return VecOfSparseMat2Arr3(asparse)
end

using SparseArrays
struct VecOfSparseMat2Arr3{Tv,Ti,V} <: AbstractArray{Tv,3}
  values::V
  function VecOfSparseMat2Arr3(values::V) where {Tv,Ti,V<:AbstractVector{<:AbstractSparseMatrix{Tv,Ti}}}
    new{Tv,Ti,V}(values)
  end
end

FEM.get_values(s::VecOfSparseMat2Arr3) = s.values
Base.size(s::VecOfSparseMat2Arr3) = (1,nnz(first(s.values)),length(s.values))

function Base.getindex(s::VecOfSparseMat2Arr3,i::Integer,j,k::Integer)
  @check i == 1
  nonzeros(s.values[k])[j]
end

function Base.getindex(s::VecOfSparseMat2Arr3,i::Integer,j,k)
  @check i == 1
  view(s,i,j,k)
end

function RB.get_nonzero_indices(s::VecOfSparseMat2Arr3)
  RB.get_nonzero_indices(first(s.values))
end

function RB.ttsvd(mat::NnzTTSnapshots{T},X::AbstractMatrix;kwargs...) where T
  N = 3
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  mat_k = copy(mat)
  C = cholesky(X)
  L = sparse(C.L)
  for k = 1:N-1
    if k == 1
      _mat_k = reshape(mat_k,:,prod(sizes[2:end]))
      mat_k = reshape(L'*_mat_k[C.p,:],ranks[k]*sizes[k],:)
    else
      mat_k = reshape(mat_k,ranks[k]*sizes[k],:)
    end
    U,Σ,V = svd(mat_k)
    rank = RB.truncation(Σ;kwargs...)
    ranks[k+1] = rank
    mat_k = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    if k == 1
      cores[k] = reshape((L'\U[:,1:rank])[invperm(C.p),:],ranks[k],sizes[k],rank)
    else
      cores[k] = reshape(U[:,1:rank],ranks[k],sizes[k],rank)
    end
  end
  return cores
end

function RB.Projection(s::NnzTTSnapshots,args...;kwargs...)
  cores = RB.ttsvd(s,args...;kwargs...)
  basis = OldTTSVDCores(cores)
  RB.recast_basis(s,basis)
end

struct OldTTSVDCores{D,A,B} <: Projection
  cores::A
  basis_spacetime::B
  function OldTTSVDCores(
    cores::A,
    basis_spacetime::B=RB.cores2basis(cores...)
    ) where {A,B}

    D = length(cores)
    new{D,A,B}(cores,basis_spacetime)
  end
end

RB.get_basis_space(b::OldTTSVDCores) = RB.cores2basis(b.cores[1:end-1]...)
RB.get_basis_time(b::OldTTSVDCores) = RB.cores2basis(b.cores[end])
RB.get_basis_spacetime(b::OldTTSVDCores) = b.basis_spacetime
RB.num_space_dofs(b::OldTTSVDCores) = prod(size.(b.cores[1:end-1],2))
RB.num_space_dofs(b::OldTTSVDCores,k::Integer) = size(b.cores[k],2)
FEM.num_times(b::OldTTSVDCores) = size(b.cores[end],2)
RB.num_reduced_space_dofs(b::OldTTSVDCores) = size(b.cores[end-1],3)
RB.num_reduced_space_dofs(b::OldTTSVDCores,k::Integer) = size(b.cores[k],3)
RB.num_reduced_times(b::OldTTSVDCores) = size(b.cores[end],3)
RB.num_fe_dofs(b::OldTTSVDCores) = RB.num_space_dofs(b)*RB.num_times(b)
RB.num_reduced_dofs(b::OldTTSVDCores) = RB.num_reduced_times(b)

function RB.recast_basis(s::NnzTTSnapshots,b::OldTTSVDCores{D}) where D
  @assert D == 2 "Spatial splitting deactivated for residuals/jacobians"
  _space_core,time_core = b.cores
  space_core = recast(s,_space_core)
  OldTTSVDCores([space_core,time_core],b.basis_spacetime)
end

function old_reduce_operator(
  mdeim_style::RB.SpaceTimeMDEIM,
  b::OldTTSVDCores,
  b_test::OldTTSVDCores;
  kwargs...)

  bs = get_basis_space(b)
  bt = get_basis_time(b)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  ns = num_reduced_space_dofs(b)
  ns_test = num_reduced_space_dofs(b_test)

  T = eltype(first(bs))
  V = Vector{T}
  b̂st = Vector{V}(undef,num_reduced_dofs(b))

  b̂s = bs_test'*bs
  cache_t = zeros(T,num_reduced_dofs(b_test))

  @inbounds for i = 1:num_reduced_dofs(b)
    bti = view(bt,:,(i-1)*ns+1:i*ns)
    for i_test = 1:num_reduced_dofs(b_test)
      ids_i = (i_test-1)*ns_test+1:i_test*ns_test
      bt_test_i = view(bt_test,:,ids_i)
      b̂ti = bt_test_i'*bti
      cache_t[i_test] = sum(b̂s .* b̂ti)
    end
    b̂st[i] = copy(cache_t)
  end

  return ReducedVectorOperator(mdeim_style,b̂st)
end

function old_reduce_operator(
  mdeim_style::RB.SpaceTimeMDEIM,
  b::OldTTSVDCores,
  b_trial::OldTTSVDCores,
  b_test::OldTTSVDCores;
  combine=(x,y)->θ*x+(1-θ)*y)

  bs = first(b.cores)
  bt = get_basis_time(b)
  bs_trial = get_basis_space(b_trial)
  bt_trial = get_basis_time(b_trial)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  ns_test = num_reduced_space_dofs(b_test)
  ns_trial = num_reduced_space_dofs(b_trial)

  T = eltype(first(bs))
  M = Matrix{T}
  b̂st = Vector{M}(undef,num_reduced_dofs(b))

  cache_t = zeros(T,num_reduced_dofs(b_test),num_reduced_dofs(b_trial))
  @inbounds for i = 1:num_reduced_dofs(b)
    b̂st[i] = copy(cache_t)
  end

  @inbounds for is = 1:num_reduced_space_dofs(b)
    b̂si = bs_test'*get_values(bs)[is]*bs_trial
    for i = 1:num_reduced_dofs(b)
      bti = view(bt,:,is)
      for i_test = 1:num_reduced_dofs(b_test), i_trial = 1:num_reduced_dofs(b_trial)
        ids_i_test = (i_test-1)*ns_test+1:i_test*ns_test
        ids_i_trial = (i_trial-1)*ns_trial+1:i_trial*ns_trial
        bti_test = view(bt_test,:,ids_i_test)
        bti_trial = view(bt_trial,:,ids_i_trial)
        b̂ti = combine_basis_time(bti,bti_trial,bti_test;combine)
        cache_t[i_test,i_trial] = sum(b̂si .* b̂ti)
      end
      b̂st[i] .+= copy(cache_t)
    end
  end

  return ReducedMatrixOperator(mdeim_style,b̂st)
end

A1_old = OldBasicTTSnapshots(A1.values,A1.realization)
mdeim_style = rbsolver.mdeim_style
old_basis = reduced_basis(A1_old;ϵ=RB.get_tol(rbsolver))
# lu_interp,integration_domain = mdeim(mdeim_style,basis)
b = RB.get_basis(get_trial(pop))
old_t_basis = OldTTSVDCores([b.cores_space...,b.core_time])
old_proj_basis = old_reduce_operator(mdeim_style,old_basis,old_t_basis,old_t_basis)

old_basis_vec = OldTTSVDCores([basis_vec.cores_space...,basis_vec.core_time])
old_proj_basis_vec = old_reduce_operator(mdeim_style,old_basis_vec,old_t_basis)
