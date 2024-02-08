using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
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
cache_solve = RB.allocate_in_range(a,r_online)
cache_recast = RB.allocate_coeff_matrix(a,r_online)

s1 = RB.tensor_getindex(s,1:ns,1:nt,1)
1 .+ (1:2 .- 1)*np

coeff = mdeim_coeff!(cache,a,b)
