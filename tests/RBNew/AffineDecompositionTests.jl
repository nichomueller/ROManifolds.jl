#############
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

np = 3
nt = 10

pranges = fill([0,1],3)
tdomain = 0:nt
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

########################## HEAT EQUATION ############################

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1)
trial = TrialFESpace(test,x->0)

ns = num_free_dofs(test)

Φ = Float64.(I(ns))
Ψ = Float64.(I(10))
red_test = RB.TestRBSpace(test,Φ,Ψ)
red_trial = RB.TrialRBSpace(trial,Φ,Ψ)

biform(u,v) = ∫(∇(v)⋅∇(u))dΩ
liform(v) = ∫(v)dΩ

_A = assemble_matrix(biform,trial,test)
_b = assemble_vector(liform,test)
snaps_biform = Snapshots(ParamArray([copy(_A) for _ = 1:nt*np]),r)
snaps_liform = Snapshots(ParamArray([copy(_b) for _ = 1:nt*np]),r)

@test norm(snaps_biform - hcat([_A.nzval for _ = 1:nt*np]...)) ≈ 0.0

np_online = 1
r_online = realization(ptspace,nparams=np_online)
snaps_biform_online = Snapshots(ParamArray([copy(_A) for _ = 1:nt]),r_online)
snaps_liform_online = Snapshots(ParamArray([copy(_b) for _ = 1:nt]),r_online)

# matrix

mdeim_style = RB.SpaceTimeMDEIM()#RB.SpaceOnlyMDEIM()
basis_space,basis_time = RB.reduced_basis(snaps_biform)
@check size(basis_space,2) == 1 && size(basis_time,2) == 1

indices_space = RB.get_mdeim_indices(basis_space)
interp_basis_space = view(basis_space,indices_space,:)
@check length(indices_space) == 1 && indices_space[1] == argmax(abs.(_A.nzval))

indices_time,lu_interp = RB._time_indices_and_interp_matrix(mdeim_style,interp_basis_space,basis_time)
if mdeim_style == RB.SpaceOnlyMDEIM()
  @check indices_time == get_times(r)
else
  @check length(indices_time) == 1 && indices_time[1] == 1
end

integration_domain = RB.ReducedIntegrationDomain(indices_space,indices_time)
proj_basis_space = map(basis_space.values) do a
  Φ'*a*Φ
end
@check basis_space.values[1] ≈ proj_basis_space[1]

comb_basis_time = zeros(Float64,size(Ψ,1),size(Ψ,2),size(Ψ,2))
@inbounds for jt = axes(Ψ,2), it = axes(Ψ,2)
  comb_basis_time[:,it,jt] .= Ψ[:,it].*Ψ[:,jt]
end

A = AffineDecomposition(
  mdeim_style,proj_basis_space,basis_time,lu_interp,integration_domain,comb_basis_time)

coeff_cache = RB.allocate_mdeim_coeff(A,r_online)
lincomb_cache = RB.allocate_mdeim_lincomb(red_trial,red_test,r_online)

snaps_mat = RB._select_snapshots_at_space_time_locations(snaps_biform_online,A,indices_time)
RB.mdeim_coeff!(coeff_cache,A,snaps_mat)
_,coeff_mat = coeff_cache
RB.mdeim_lincomb!(lincomb_cache,A,coeff_mat)
_,_,mat_red = lincomb_cache

red_mat_snaps = RB.compress(red_trial,red_test,snaps_biform_online)
err = red_mat_snaps - mat_red[1]

@check norm(err)/sqrt(length(err)) ≤ 1e-12

# vector

mdeim_style = RB.SpaceTimeMDEIM() # RB.SpaceOnlyMDEIM() #
basis_space,basis_time = RB.reduced_basis(snaps_liform)
@check size(basis_space,2) == 1 && size(basis_time,2) == 1

indices_space = RB.get_mdeim_indices(basis_space)
interp_basis_space = view(basis_space,indices_space,:)
@check length(indices_space) == 1 && indices_space[1] == argmax(abs.(_b))

indices_time,lu_interp = RB._time_indices_and_interp_matrix(mdeim_style,interp_basis_space,basis_time)
if mdeim_style == RB.SpaceOnlyMDEIM()
  @check indices_time == get_times(r)
else
  @check length(indices_time) == 1 && indices_time[1] == 1
end

integration_domain = RB.ReducedIntegrationDomain(indices_space,indices_time)
proj_basis_space = map(eachcol(basis_space)) do a
  Φ'*a
end
@check basis_space ≈ proj_basis_space[1]

comb_basis_time = Ψ

b = AffineDecomposition(
  mdeim_style,proj_basis_space,basis_time,lu_interp,integration_domain,comb_basis_time)

coeff_cache = RB.allocate_mdeim_coeff(b,r_online)
lincomb_cache = RB.allocate_mdeim_lincomb(red_test,r_online)

snaps_vec = RB._select_snapshots_at_space_time_locations(snaps_liform_online,b,indices_time)
@check all(abs.(snaps_vec) .≈ maximum(abs.(snaps_liform_online)))

RB.mdeim_coeff!(coeff_cache,b,snaps_vec)
_,coeff_vec = coeff_cache
RB.mdeim_lincomb!(lincomb_cache,b,coeff_vec)
_,_,vec_red = lincomb_cache

red_vec_snaps = RB.compress(red_test,snaps_liform_online)
err = red_vec_snaps - vec_red[1]

@check norm(err)/sqrt(length(err)) ≤ 1e-12
