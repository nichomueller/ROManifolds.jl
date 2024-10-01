using Gridap
using Gridap.FESpaces
using Gridap.Algebra
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
dt = 0.0025
t0 = 0.0
tf = 0.15

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 12
domain = (0,1,0,1/3,0,1/4)
partition = (n,floor(n/3),floor(n/4))
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet0",setdiff(1:26,22))
add_tag_from_tags!(labels,"neumann",[22])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model.model,tags="neumann")
dΓn = Measure(Γn,degree)

a(x,μ,t) = 1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = VectorValue(1/μ[1],0.0,0.0)
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g0(μ,t) = x->g0(x,μ,t)
g0μt(μ,t) = TransientParamFunction(g0,μ,t)

h(x,μ,t) = VectorValue(μ[1]*abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100),0.0,0.0)
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

const Re = 100
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res_lin(μ,t,(u,p),(v,q),dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ) - ∫(v⋅fμt(μ,t))dΩ - ∫(v⋅hμt(μ,t))dΓn

res_nlin(μ,t,(u,p),(v,q),dΩ,dΓn) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,g0μt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop_lin = TransientParamLinearFEOperator((stiffness,mass),res_lin,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
# feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,
#   trial,test,trian_res,trian_stiffness,trian_mass)
# feop = LinNonlinTransientParamFEOperator(feop_lin,feop_nlin)
feop = feop_lin

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(NewtonRaphsonSolver(LUSolver(),1e-10,20),dt,θ)

energy_u(u,v) = ∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ

stiffness_u(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ
mass_u(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res_u(μ,t,u,v,dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness_u(μ,t,u,v,dΩ) - ∫(v⋅hμt(μ,t))dΓn

res_nlin_u(μ,t,u,v,dΩ) = c(u,v,dΩ)
jac_nlin_u(μ,t,u,du,v,dΩ) = dc(u,du,v,dΩ)

feop_lin_u = TransientParamLinearFEOperator((stiffness_u,mass_u),res_u,ptspace,
  trial_u,test_u,trian_res,trian_stiffness,trian_mass)
# feop_nlin_u = TransientParamFEOperator(res_nlin_u,jac_nlin_u,ptspace,
#   trial_u,test_u,trian_stiffness,trian_stiffness)
# feop_u = LinNonlinTransientParamFEOperator(feop_lin_u,feop_nlin_u)
feop_u = feop_lin_u

tol = fill(1e-4,5)
reduction = TTSVDReduction(tol,energy_u;nparams=50)
rbsolver = RBSolver(fesolver,reduction;nparams_res=30,nparams_jac=20,nparams_djac=0)

# r = realization(feop;nparams=60)
# fesnaps,festats = solution_snapshots(rbsolver,feop,r,xh0μ)
# roffline = r[1:50,:]
# ronline = r[51:60,:]
# fesnaps_off = select_snapshots(fesnaps[1],1:50)
# fesnaps_on = select_snapshots(fesnaps[1],51:60)

test_dir = datadir("temp_stokes_folder")
create_dir(test_dir)

# save(test_dir,fesnaps[1])
fesnaps = load_snapshots(test_dir)
r = get_realization(fesnaps)
roffline = r[1:50,:]
ronline = r[51:60,:]
fesnaps_off = select_snapshots(fesnaps,1:50)
fesnaps_on = select_snapshots(fesnaps,51:60)

rbop = reduced_operator(rbsolver,feop_u,fesnaps_off)
save(test_dir,rbop)
# rbop = load_operator(test_dir,feop_u)

x̂,rbstats = solve(rbsolver,rbop,ronline)
# perf = rb_performance(rbsolver,rbop,fesnaps_on,x̂,festats,rbstats,ronline)
perf = rb_performance(rbsolver,rbop,fesnaps_on,x̂,rbstats,rbstats,ronline)

println(perf)

# red_test,red_trial = reduced_fe_space(rbsolver,feop_u,fesnaps_off)
# op = get_algebraic_operator(feop_nlin_u)
# jacs = jacobian_snapshots(rbsolver,op,fesnaps_off)
# jac_red = RBSteady.get_jacobian_reduction(rbsolver)[1]
# # red_jac = reduced_jacobian(jac_red,red_trial,red_test,jac)
# red = get_reduction(jac_red)
# basis = projection(red,jacs[1][1])
# proj_basis = project(red_test,basis,red_trial,jac_red.combine)

# cores_test = red_test.subspace.cores
# cores = basis.cores
# cores_trial = cores_test
# # contraction(cores_test[3],cores[3],cores_trial[3])

# factor1,factor2,factor3 = cores_test[3],cores[3],cores_trial[3]
# A = reshape(permutedims(factor1,(1,3,2)),:,size(factor1,2))
# B = reshape(permutedims(factor2,(1,3,2)),:,size(factor2,2))
# C = reshape(permutedims(factor3,(2,1,3)),size(factor3,2),:)
# sparsity = factor2.sparsity
# BC = RBSteady._sparsemul(B,C,sparsity)
# ABC = A*BC

using Mabla.FEM.IndexMaps
s1 = select_snapshots(change_index_map(TrivialIndexMap,fesnaps[1]),1)
r1 = get_realization(s1)
U1 = trial_u(r1)
using Gridap.Visualization
dir = datadir("plts")
createpvd(dir) do pvd
  for i in param_eachindex(r1)
    file = dir*"/u$i"*".vtu"
    Ui = param_getindex(U1,i)
    vi = s1[:,i,1]
    uhi = FEFunction(Ui,vi)
    pvd[i] = createvtk(Ω.trian,file,cellfields=["u"=>uhi])
  end
end
