using Gridap
using Gridap.FESpaces
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using Mabla.FEM
using Mabla.FEM.Utils
using Mabla.FEM.IndexMaps
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

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

n = 5
domain = (0,1,0,1)
partition = (n,n)
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet0",[1,2,3,4,5,7,8])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model.model,tags=[6])
dΓn = Measure(Γn,degree)

a(x,μ,t) = 1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

g0(x,μ,t) = VectorValue(0.0,0.0)
g0(μ,t) = x->g0(x,μ,t)
g0μt(μ,t) = TransientParamFunction(g0,μ,t)

f1(x,μ,t) = VectorValue(x[2]*(1-x[2])*abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100),0.0)
f2(x,μ,t) = VectorValue(0.0,x[1]*(1-x[1])*abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100))
f(x,μ,t) = f1(x,μ,t) + f2(x,μ,t)
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = VectorValue(abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100),0.0)
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ) - ∫(v⋅hμt(μ,t))dΓn #- ∫(v⋅fμt(μ,t))dΩ

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,g0μt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

# energy_u(u,v) = ∫(v*u)dΩ + ∫(∇(v)⊙∇(u))dΩ

# tol = fill(1e-4,4)
# reduction = TTSVDReduction(tol,energy_u;nparams=20)
# rbsolver = RBSolver(fesolver,reduction;nparams_test=2,nparams_res=10,nparams_jac=10)
# fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)

# X = assemble_matrix(feop,energy)
# X1 = X[1]
# s1 = select_snapshots(fesnaps[1],RBSteady.offline_params(rbsolver))
# cores = reduced_basis(reduction,s1,X1)

# stiffness_u(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ
# mass_u(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
# res_u(μ,t,u,v,dΩ) = ∫(v⋅∂t(u))dΩ + stiffness_u(μ,t,u,v,dΩ) - ∫(v⋅fμt(μ,t))dΩ

# feop_u = TransientParamLinearFEOperator((stiffness_u,mass_u),res_u,ptspace,
#   trial_u,test_u,trian_res,trian_stiffness,trian_mass)

# op = get_algebraic_operator(feop_u)
# red_test = TransientRBSpace(test_u,cores)
# red_trial = TransientRBSpace(trial_u,cores)
# pop = TransientPGOperator(op,red_trial,red_test)
# jacs,ress = jacobian_and_residual(rbsolver,pop,fesnaps[1])

# red_jac = reduced_jacobian(RBSteady.get_jacobian_reduction(rbsolver),pop,jacs)
# red_res = reduced_residual(RBSteady.get_residual_reduction(rbsolver),pop,ress)

# using Gridap.CellData
# trians_rhs = get_domains(red_res)
# trians_lhs = map(get_domains,red_jac)
# new_op = change_triangulation(pop,trians_rhs,trians_lhs)
# rbop = TransientPGMDEIMOperator(new_op,red_jac,red_res)

# rbsnaps,stats,cache = solve(rbsolver,rbop,fesnaps[1])
# results = rb_performance(rbsolver,rbop,fesnaps[1],rbsnaps,festats,stats)

energy_u(u,v) = ∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ

stiffness_u(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ
mass_u(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res_u(μ,t,u,v,dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness_u(μ,t,u,v,dΩ) - ∫(v⋅hμt(μ,t))dΓn #- ∫(v⋅fμt(μ,t))dΩ

feop_u = TransientParamLinearFEOperator((stiffness_u,mass_u),res_u,ptspace,
  trial_u,test_u,trian_res,trian_stiffness,trian_mass)

tol = fill(1e-4,5)
reduction = TTSVDReduction(tol,energy_u;nparams=50)
rbsolver = RBSolver(fesolver,reduction;nparams_test=10,nparams_res=30,nparams_jac=20)
fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)

rbop = reduced_operator(rbsolver,feop_u,fesnaps[1])
rbsnaps,rbstats,cache = solve(rbsolver,rbop,fesnaps)
results = rb_performance(rbsolver,rbop,get_component(fesnaps[1],1),get_component(rbsnaps,1),festats,rbstats)

println(results)

# WITH TPOD

model = model.model
Ω = Ω.trian
dΩ = dΩ.measure

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ) - ∫(v⋅hμt(μ,t))dΓn# - ∫(v⋅fμt(μ,t))dΩ

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,g0μt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

energy_u(u,v) = ∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ

stiffness_u(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ
mass_u(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res_u(μ,t,u,v,dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness_u(μ,t,u,v,dΩ) - ∫(v⋅hμt(μ,t))dΓn

feop_u = TransientParamLinearFEOperator((stiffness_u,mass_u),res_u,ptspace,
  trial_u,test_u,trian_res,trian_stiffness,trian_mass)

tol = 1e-4
# reduction = TransientPODReduction(tol,energy;nparams=50)
reduction_u = TransientPODReduction(tol,energy_u;nparams=50)
# rbsolver = RBSolver(fesolver,reduction;nparams_test=10,nparams_res=50,nparams_jac=50)
rbsolver_u = RBSolver(fesolver,reduction_u;nparams_test=10,nparams_res=50,nparams_jac=50)

_fesnaps,_festats = solution_snapshots(rbsolver,feop,xh0μ;r=get_realization(fesnaps))

# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats,cache = solve(rbsolver,rbop,fesnaps)
# results = rb_performance(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

rbop_u = reduced_operator(rbsolver_u,feop_u,_fesnaps[1])
rbsnaps_u,rbstats_u,_ = solve(rbsolver,rbop_u,_fesnaps[1])
results_u = rb_performance(rbsolver_u,rbop_u,_fesnaps[1],rbsnaps_u,_festats,rbstats_u)

println(results_u)

# COMPARISON

X = assemble_matrix(feop_u,energy_u)
bs = get_basis_space(rbop.op.test.basis)
_bs = get_basis_space(rbop_u.op.test.basis)

bs'*X*bs
_bs'*X*_bs

idom = rbop.lhs[1][1].integration_domain
_idom = rbop_u.lhs[1][1].integration_domain

b = rbop.lhs[1][1].basis.basis
_b = rbop_u.lhs[1][1].basis.basis

idom = rbop.rhs[1].integration_domain
_idom = rbop_u.rhs[1].integration_domain
idom = rbop.rhs[2].integration_domain
_idom = rbop_u.rhs[2].integration_domain


function _jacobian(solver::RBSolver,op,s)
  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,RBSteady.jac_params(solver))
  us_jac = (get_values(sjac),)
  r_jac = get_realization(sjac)
  A = jacobian(fesolver,op,r_jac,us_jac)
  iA = get_matrix_index_map(op)
  return Snapshots(A,iA,r_jac)
end

op = get_algebraic_operator(feop_u)
jacs = _jacobian(rbsolver,op,fesnaps[1])

basis = reduced_basis(rbsolver.jacobian_reduction[1].reduction,jacs[1][1])
# red = rbsolver.jacobian_reduction[1].reduction
# s = jacs[1][1]
# cores_space...,core_time = reduction(red,s)
# cores_space′ = recast(s,cores_space)
# index_map = get_index_map(s)

c1 = compress_core(basis.cores_space[1],b.cores_space[1],b.cores_space[1])
c2 = compress_core(basis.cores_space[2],b.cores_space[2],b.cores_space[2])
cc = compress_core(basis.cores_space[3],b.cores_space[3],b.cores_space[3])

s1 = select_snapshots(fesnaps[1],51)
sa1 = select_snapshots(rbsnaps,1)
e1 = abs.(s1 - sa1)
r1 = get_realization(s1)
U1 = trial_u(r1)

using Gridap.Visualization
dir = datadir("plts")
createpvd(dir) do pvd
  for i in param_eachindex(r1)
    file = dir*"/u$i"*".vtu"
    Ui = param_getindex(U1,i)
    vi = sa1[:,i,1]
    uhi = FEFunction(Ui,vi)
    pvd[i] = createvtk(Ω,file,cellfields=["u"=>uhi])
  end
end


# s1 = select_snapshots(fesnaps[1],51)
# sa1 = select_snapshots(rbsnaps,1)
# r1 = get_realization(s1)
# U1 = trial_u(r1)

# for i in param_eachindex(r1)
#   vi = s1[:,i,1]
#   Ui = param_getindex(U1,i)
#   uhi = FEFunction(Ui,vi)
#   println( norm( assemble_vector(p -> ∫(p*(∇⋅(uhi)))dΩ,test_p) ) )
# end

# for i in param_eachindex(r1)
#   vi = sa1[:,i,1]
#   Ui = param_getindex(U1,i)
#   uhi = FEFunction(Ui,vi)
#   println( norm( assemble_vector(p -> ∫(p*(∇⋅(uhi)))dΩ,test_p) ) )
# end
