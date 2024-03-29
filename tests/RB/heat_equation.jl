using Gridap
using Test
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)

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

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=1,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver,dir=test_dir)

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

# POD-MDEIM error
pod_err,mdeim_error = RB.pod_mdeim_error(rbsolver,feop,rbop,fesnaps)

ϵ = 1e-4
rbsolver_space = RBSolver(fesolver,ϵ,RB.SpaceOnlyMDEIM();nsnaps_state=50,nsnaps_test=1,nsnaps_mdeim=20)
test_dir_space = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver_space,dir=test_dir_space)

rbop_space = reduced_operator(rbsolver_space,feop,fesnaps)
rbsnaps_space,rbstats_space = solve(rbsolver_space,rbop,fesnaps)
results_space = rb_results(rbsolver_space,feop,fesnaps,rbsnaps_space,festats,rbstats_space)

println(RB.space_time_error(results_space))
save(test_dir,rbop_space)
save(test_dir,results_space)

# #################
# using Gridap.FESpaces
# red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# odeop = get_algebraic_operator(feop)
# pop = PODOperator(odeop,red_trial,red_test)
# smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
# jjac,rres = jacobian_and_residual(rbsolver,pop,smdeim)

# # jacobian
# sjac = jjac[1][1]
# mdeim_style = rbsolver.mdeim_style
# basis = reduced_basis(sjac;ϵ=RB.get_tol(rbsolver))
# lu_interp,integration_domain = mdeim(mdeim_style,basis)
# # proj_basis = reduce_operator(mdeim_style,basis,get_trial(pop),get_test(pop))

# bs = get_basis_space(basis)
# bt = get_basis_time(basis)
# b_trial = RB.get_basis(get_trial(pop))
# b_test = RB.get_basis(get_test(pop))
# bs_trial = get_basis_space(b_trial)
# bt_trial = get_basis_time(b_trial)
# bs_test = get_basis_space(b_test)
# bt_test = get_basis_time(b_test)

# M = Matrix{eltype(bs)}
# b̂st = Vector{M}(undef,num_reduced_dofs(basis))
# combine = (x,y) -> θ*x+(1-θ)*y

# b̂t = combine_basis_time(bt,bt_trial,bt_test;combine)

# @inbounds for is = 1:num_reduced_space_dofs(basis)
#   b̂si = bs_test'*get_values(bs)[is]*bs_trial
#   for it = 1:num_reduced_times(basis)
#     ist = (it-1)*num_reduced_space_dofs(basis)+is
#     b̂ti = b̂t[it]
#     b̂st[ist] = kronecker(b̂ti,b̂si)
#   end
# end

# online stuff
using Gridap.FESpaces
using Gridap.Algebra
using Gridap.ODEs
son = select_snapshots(fesnaps,RB.online_params(rbsolver))
ron = get_realization(son)
fesolver = RB.get_fe_solver(rbsolver)
fe_trial = get_fe_trial(pop)(ron)
x̂ = zero_free_values(red_trial)
y = zero_free_values(fe_trial)
odecache = allocate_odecache(fesolver,pop,ron,(y,))
_,odeopcache = odecache
FEM.shift!(ron,dt*(θ-1))

dtθ = dt*θ
ws = (1,1/dtθ)
Acache = allocate_jacobian(pop,ron,(y,y),odeopcache)
fe_sA = fe_jacobian!(Acache,rbop,ron,(y,y),ws,odeopcache)
# Â = RB.mdeim_jacobian(rbop.lhs,fe_sA)

ad = rbop.lhs[1][1]
RB.coefficient!(ad,fe_sA[1][1])
basis = ad.basis
coefficient = ad.coefficient
result = ad.result
fill!(result,zero(eltype(result)))

i = 1
# basis*coefficient[i]
contrib1 = basis.basis[1]*coefficient[i][1]
contrib2 = basis.basis[3]*coefficient[i][2]
