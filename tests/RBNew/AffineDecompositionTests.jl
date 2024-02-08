using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using SparseArrays
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using DrWatson
using Mabla.FEM
using Mabla.RB

ns = 100
nt = 10
np = 5
pranges = fill([0,1],3)
tdomain = 0:1:10
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)

### vector

# simulate the RB
v = [rand(ns) for i = 1:np*nt]
a = ParamArray(v)
s = Snapshots(a,r)
Φ,Ψ = RB.reduced_basis(s)

# snaps to compress
v = [rand(ns) for i = 1:np*nt]
a = ParamArray(v)
s = Snapshots(a,r)

mdeim_style = RB.SpaceOnlyMDEIM()
basis_space,basis_time = RB.reduced_basis(s;rank=10)
indices_space = RB.get_mdeim_indices(basis_space)
interp_basis_space = view(basis_space,indices_space,:)
indices_time,lu_interp = RB._time_indices_and_interp_matrix(mdeim_style,interp_basis_space,basis_time)
rindices_space = RB.recast_indices_space(basis_space,indices_space)
integration_domain = RB.ReducedIntegrationDomain(rindices_space,indices_time)
proj_basis_space = map(eachcol(basis_space)) do a
  Φ'*a
end
comb_basis_time = Ψ

a = AffineDecomposition(
  mdeim_style,proj_basis_space,basis_time,lu_interp,integration_domain,comb_basis_time)

np_online = 1
r_online = realization(ptspace,nparams=np_online)
cache_solve = RB.allocate_coeff_matrix(a,r_online)
cache_recast = RB.allocate_param_coeff_matrix(a,r_online)
coeff_cache = cache_solve,cache_recast

_s = RB.InnerTimeOuterParamTransientSnapshots(s)
b = stack(RB.tensor_getindex(_s,:,:,1))
red_b = vec(Φ'*b*Ψ)
bi = b[indices_space,:]

RB.mdeim_coeff!(coeff_cache,a,b)
_,coeff = coeff_cache

V = Vector{Float64}
ns_test = size(Φ,2)
nt_test = size(Ψ,2)
time_prod_cache = allocate_vector(V,nt_test)
kron_prod_cache = allocate_vector(V,ns_test*nt_test)
lincomb_cache = allocate_param_array(kron_prod_cache,num_params(r_online))
lcache = time_prod_cache,kron_prod_cache,lincomb_cache

RB.mdeim_lincomb!(lcache,a,coeff)
_,_,mdeim_approx = lcache

red_b - mdeim_approx[1]

######## matrix

# simulate the RB
v = [rand(ns) for i = 1:np*nt]
a = ParamArray(v)
s = Snapshots(a,r)
Φ,Ψ = RB.reduced_basis(s)

# snaps to compress
v = [sparse(rand(ns,ns)) for i = 1:np*nt]
a = ParamArray(v)
s = Snapshots(a,r)

mdeim_style = RB.SpaceOnlyMDEIM()
basis_space,basis_time = RB.reduced_basis(s;rank=10)
indices_space = RB.get_mdeim_indices(basis_space)
interp_basis_space = view(basis_space,indices_space,:)
indices_time,lu_interp = RB._time_indices_and_interp_matrix(mdeim_style,interp_basis_space,basis_time)
rindices_space = RB.recast_indices_space(basis_space,indices_space)
integration_domain = RB.ReducedIntegrationDomain(rindices_space,indices_time)
proj_basis_space = map(basis_space.values) do a
  Φ'*a*Φ
end

comb_basis_time = zeros(Float64,size(Ψ,1),size(Ψ,2),size(Ψ,2))
@inbounds for jt = axes(Ψ,2), it = axes(Ψ,2)
  comb_basis_time[:,it,jt] .= Ψ[:,it].*Ψ[:,jt]
end

A = AffineDecomposition(
  mdeim_style,proj_basis_space,basis_time,lu_interp,integration_domain,comb_basis_time)

np_online = 1
r_online = realization(ptspace,nparams=np_online)
cache_solve = RB.allocate_coeff_matrix(A,r_online)
cache_recast = RB.allocate_param_coeff_matrix(A,r_online)
coeff_cache = cache_solve,cache_recast

_s = RB.InnerTimeOuterParamTransientSnapshots(s)
M = view(_s,:,:,1)
red_b = vec(Φ'*b*Ψ)
Mi = M[indices_space,:]

RB.mdeim_coeff!(coeff_cache,A,Mi)
_,coeff = coeff_cache

V = Vector{Float64}
ns_test = size(Φ,2)
nt_test = size(Ψ,2)
time_prod_cache = allocate_vector(V,nt_test)
kron_prod_cache = allocate_vector(V,ns_test*nt_test)
lincomb_cache = allocate_param_array(kron_prod_cache,num_params(r_online))
lcache = time_prod_cache,kron_prod_cache,lincomb_cache

RB.mdeim_lincomb!(lcache,A,coeff)
_,_,mdeim_approx = lcache

red_A - mdeim_approx[1]
