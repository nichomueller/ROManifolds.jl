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

# red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# odeop = get_algebraic_operator(feop)
# pop = PODOperator(odeop,red_trial,red_test)
# red_lhs,red_rhs = reduced_jacobian_residual(rbsolver,pop,fesnaps)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)

son = select_snapshots(fesnaps,RB.online_params(rbsolver))
r = get_realization(son)
trial = get_trial(rbop)(r)
fe_trial = get_fe_trial(rbop)(r)
x̂ = zero_free_values(trial)
y = zero_free_values(fe_trial)

odecache = allocate_odecache(fesolver,rbop,r,(y,))
odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache

x = copy(y)
fill!(x,zero(eltype(x)))
dtθ = θ*dt
shift!(r,dt*(θ-1))
us = (x,x)
ws = (1,1/dtθ)

update_odeopcache!(odeopcache,rbop,r)

# stageop = LinearParamStageOperator(rbop,odeopcache,r,us,ws,A,b,reuse,sysslvrcache)
# bnew = residual!(b,rbop,r,us,odeopcache)
# fe_sb = fe_residual!(b,rbop,r,us,odeopcache)
# b̂ = mdeim_result(rbop.rhs,fe_sb)
# aa,bb = rbop.rhs[1],fe_sb[1]
# RB.coefficient!(aa,bb)
# basis = aa.basis
# coefficient = aa.coefficient
# result = aa.result
# fill!(result,zero(eltype(result)))
# @inbounds for i = eachindex(result)
#   result[i] .= basis*coefficient[i]
# end

# red_r,red_times,red_us,red_odeopcache = RB._select_fe_quantities_at_time_locations(rbop.rhs,r,us,odeopcache)
# bb = residual!(b,rbop.op,red_r,red_us,red_odeopcache)
# # bi = RB._select_snapshots_at_space_time_locations(bb,rbop.rhs,red_times)
# ss,aa = bb[1],rbop.rhs[1]
# ids_space = RB.get_indices_space(aa)
# ids_time = filter(!isnothing,indexin(RB.get_indices_time(aa),red_times))
# srev = reverse_snapshots(ss)
# RB.select_snapshots_entries(srev,ids_space,ids_time)

# fe_sA = fe_jacobian!(A,rbop,r,us,ws,odeopcache)
red_r,red_times,red_us,red_odeopcache = RB._select_fe_quantities_at_time_locations(rbop.lhs,r,us,odeopcache)
AA = jacobian!(A,rbop.op,red_r,red_us,ws,red_odeopcache)
AA11 = AA[1][1]
# Â = mdeim_result(rbop.lhs,fe_sA)

#
soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
red_trial,red_test = reduced_fe_space(rbsolver,feop,soff)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
A,b = jacobian_and_residual(rbsolver,pop,smdeim)
mdeim_style = rbsolver.mdeim_style

A1,t1 = A[1].values[1],A[1].trians[1]
# A1,t1 = b.values[1],b.trians[1]
basis = reduced_basis(A1;ϵ=RB.get_tol(rbsolver))
basis_spacetime = get_basis_spacetime(basis)
indices_spacetime = get_mdeim_indices(basis_spacetime)
indices_space = fast_index(indices_spacetime,RB.num_space_dofs(basis))
indices_time = slow_index(indices_spacetime,RB.num_space_dofs(basis))
recast_indices_space = RB.recast_indices(basis,indices_space)

space_dofs = RB._num_tot_space_dofs(basis)
tensor_indices = RB.tensorize_indices(vec(prod(space_dofs;dims=1)),indices)
RB._split_row_col(space_dofs,tensor_indices)
# indices = [22]
# dofs = [25,16]#RB._num_tot_space_dofs(basis)
# D = length(dofs)
# cdofs = cumprod(dofs)
# tindices = Vector{CartesianIndex{D}}(undef,length(indices))
# ii,i = 1,22
# ic = ()
# @inbounds for d = 1:D-1
#   ic = (ic...,fast_index(i,cdofs[d]))
# end
# ic = (ic...,slow_index(i,cdofs[D-1]))
# tindices[ii] = CartesianIndex(ic)

M = reshape(basis_space,20,20)
p = free_dofs_map(test.dof_permutation)
p̃ = test.tp_dof_permutation

Mpp = M[p̃[:],p̃[:]][invperm(p[:]),invperm(p[:])]

iis = [22]
cids = CartesianIndices(RB.num_space_dofs(A1))
cispace = map(i->getindex(cids,i),iis)
A1[iis]

aa,bb = basis.cores_space
boh = kronecker(bb[1,:,:,1],aa[1,:,:,1])

row,col = fast_index(22,20),slow_index(22,20)
idsx = CartesianIndex(fast_index(row,5),fast_index(col,5))
idsy = CartesianIndex(slow_index(row,5),slow_index(col,5))
# rows = CartesianIndex(fast_index(row,5),slow_index(row,5))
# cols = CartesianIndex(fast_index(col,5),slow_index(col,5))
