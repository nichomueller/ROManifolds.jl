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

const Re = 100
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ) - ∫(v⋅hμt(μ,t))dΓn

res_nlin(μ,t,(u,p),(v,q),dΩ,dΓn) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

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
feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
feop = LinNonlinTransientParamFEOperator(feop_lin,feop_nlin)

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
feop_nlin_u = TransientParamFEOperator(res_nlin_u,jac_nlin_u,ptspace,
  trial_u,test_u,trian_stiffness,trian_stiffness)
feop_u = LinNonlinTransientParamFEOperator(feop_lin_u,feop_nlin_u)

tol = fill(1e-3,5)
reduction = TTSVDReduction(tol,energy_u;nparams=50)
rbsolver = RBSolver(fesolver,reduction;nparams_res=30,nparams_jac=20,nparams_djac=0)

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
ronline = realization(feop_u;nparams=10)
x,festats = solution_snapshots(rbsolver,feop,ronline,xh0μ)
fesnaps_u = fesnaps[1]

rbop = reduced_operator(rbsolver,feop_u,fesnaps_u)
x̂,rbstats = solve(rbsolver,rbop,ronline)
perf = rb_performance(rbsolver,rbop,x[1],x̂,festats,rbstats,ronline)

test_dir = datadir("temp_folder")
create_dir(test_dir)

save(test_dir,fesnaps_u)
save(test_dir,rbop)

fesnaps_u = load_snapshots(test_dir)
rbop = load_operator(test_dir,feop_u)

ronline = realization(feop_u;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,ronline)
perf = rb_performance(rbsolver,rbop,x[1],x̂,festats,rbstats,ronline)

load_operator(test_dir,feop_u.op_linear)

################################################################################


################################################################################
using Gridap.FESpaces
using LinearAlgebra
using PartitionedArrays

A = fesnaps_u
red = rbsolver.state_reduction
red_style = ReductionStyle(red)
X = assemble_matrix(feop_u,energy_u)

K = rank(X)
cores_k,remainders_k = map(k -> RBSteady.steady_ttsvd(red_style,A,X[k]),1:K) |> tuple_of_arrays

# ##############
# U1 = select_snapshots(A,1:25)
# U2 = select_snapshots(A,26:50)

# c1,r1 = ttsvd(red_style,U1)
# l2cores1 = [c1...,r1]
# c2,r2 = ttsvd(red_style,U2)
# l2cores2 = [c2...,r2]

# c2b1 = cores2basis(l2cores1...)
# c2b2 = cores2basis(l2cores2...)

# maximum(abs.(c2b1 - U1[:]))
# maximum(abs.(c2b2 - U2[:]))

# l2cores = RBSteady.block_cores(l2cores1,l2cores2)
# l2cores[end] = RBSteady.last_block(l2cores1[end],l2cores2[end])
# c2b = cores2basis(l2cores...)
# U = U1+U2

# maximum(abs.(c2b - U[:]))

# l2c1,r1 = RBSteady.reduce_rank(l2cores[1])
# l2cores[2] = RBSteady.absorb(l2cores[2],r1)
# l2c2,r2 = RBSteady.reduce_rank(l2cores[2])
# ##############

# # TRIAL 1
# cores = RBSteady.block_cores(cores_k...)
# remainder = cat(remainders_k...;dims=1)
# c1,R1 = RBSteady.reduce_rank(red_style[1],cores[1])
# newcore2 = reshape(R1*reshape(cores[2],size(cores[2],1),:),:,size(cores[2],2),size(cores[2],3))
# c2,R2 = RBSteady.reduce_rank(red_style[2],newcore2)

# # TRIAL 2
# cores_11 = cores_k[1][1]
# cores_21 = cores_k[2][1]
# L11,p11 = RBSteady._cholesky_decomp(X[1][1])
# L21,p21 = RBSteady._cholesky_decomp(X[2][1])
# cores_11′ = reshape(L11'*reshape(cores_11,:,size(cores_11,3))[p11,:],size(cores_11))
# cores_21′ = reshape(L21'*reshape(cores_21,:,size(cores_21,3))[p21,:],size(cores_21))
# cores1 = RBSteady.first_block(cores_11′,cores_21′)
# c1_new,R1_new = RBSteady.reduce_rank(red_style[1],cores1)

# cores_12 = cores_k[1][2]
# cores_22 = cores_k[2][2]
# L12,p12 = RBSteady._cholesky_decomp(X[1][2])
# L12′ = kron(I(size(cores_12,1)),L12)
# p12′ = vec(((collect(1:size(cores_12,1)).-1)*size(cores_12,2) .+ p12')')
# L22,p22 = RBSteady._cholesky_decomp(X[2][2])
# L22′ = kron(I(size(cores_22,1)),L22)
# p22′ = vec(((collect(1:size(cores_22,1)).-1)*size(cores_22,2) .+ p22')')
# cores_12′ = reshape(L12′'*reshape(cores_12,:,size(cores_12,3))[p12′,:],size(cores_12))
# cores_22′ = reshape(L22′'*reshape(cores_22,:,size(cores_22,3))[p22′,:],size(cores_22))
# cores2 = RBSteady.block_core(cores_12′,cores_22′)
# cores2′ = reshape(R1_new*reshape(cores2,size(cores2,1),:),:,size(cores2,2),size(cores2,3))
# c2_new,R2_new = RBSteady.reduce_rank(red_style[2],cores2′)

# ####
# L1,p1 = RBSteady._cholesky_decomp(X[1][1])
# U1,S1,V1 = RBSteady.tpod(red_style[1],reshape(A,size(A,1),:),L1,p1)
# ################################################################################

tol4 = fill(1e-4,5)
reduction4 = TTSVDReduction(tol4,energy_u;nparams=50)
rbsolver4 = RBSolver(fesolver,reduction4;nparams_res=30,nparams_jac=20,nparams_djac=0)

rbop4 = reduced_operator(rbsolver4,feop_u,fesnaps_u)
x̂4,rbstats4 = solve(rbsolver4,rbop4,ronline)
perf4 = rb_performance(rbsolver4,rbop4,x[1],x̂4,festats,rbstats4,ronline)

tol2 = fill(1e-2,5)
reduction2 = TTSVDReduction(tol2,energy_u;nparams=50)
rbsolver2 = RBSolver(fesolver,reduction2;nparams_res=30,nparams_jac=20,nparams_djac=0)

rbop2 = reduced_operator(rbsolver2,feop_u,fesnaps_u)
x̂2,rbstats2 = solve(rbsolver2,rbop2,ronline)
perf2 = rb_performance(rbsolver2,rbop2,x[1],x̂2,festats,rbstats2,ronline)

#

model′ = model.model
Ω′ = Ω.trian
dΩ′ = dΩ.measure

test_u′ = TestFESpace(model′,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
trial_u′ = TransientTrialParamFESpace(test_u′,g0μt)
test_p′ = TestFESpace(model′,reffe_p;conformity=:C0)
trial_p′ = TrialFESpace(test_p′)
test′ = TransientMultiFieldParamFESpace([test_u′,test_p′];style=BlockMultiFieldStyle())
trial′ = TransientMultiFieldParamFESpace([trial_u′,trial_p′];style=BlockMultiFieldStyle())
feop_lin′ = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial′,test′,trian_res,trian_stiffness,trian_mass)
feop_nlin′ = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,
  trial′,test′,trian_res,trian_stiffness,trian_mass)
feop′ = LinNonlinTransientParamFEOperator(feop_lin′,feop_nlin′)

feop_lin_u′ = TransientParamLinearFEOperator((stiffness_u,mass_u),res_u,ptspace,
  trial_u′,test_u′,trian_res,trian_stiffness,trian_mass)
feop_nlin_u′ = TransientParamFEOperator(res_nlin_u,jac_nlin_u,ptspace,
  trial_u′,test_u′,trian_stiffness,trian_stiffness)
feop_u′ = LinNonlinTransientParamFEOperator(feop_lin_u′,feop_nlin_u′)

xh0μ′(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial′(μ,t0))

energy_u′(u,v) = ∫(v⋅u)dΩ′ + ∫(∇(v)⊙∇(u))dΩ′

tol = 1e-4
reduction′ = TransientReduction(tol,energy_u′;nparams=50)
rbsolver′ = RBSolver(fesolver,reduction′;nparams_res=30,nparams_jac=20,nparams_djac=0)

r = get_realization(fesnaps_u)
fesnaps′,festats = solution_snapshots(rbsolver′,feop′,r,xh0μ′)
fesnaps_u′ = fesnaps′[1]
rbop′ = reduced_operator(rbsolver′,feop_u′,fesnaps_u′)
x̂,rbstats = solve(rbsolver,rbop,ronline)

x,festats = solution_snapshots(rbsolver,feop,ronline,xh0μ)
perf = rb_performance(rbsolver,rbop,x[1],x̂,festats,rbstats,ronline)
