using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapGmsh

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

using Mabla.FEM
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
tf = 0.5

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

msh_file = datadir("meshes/perforated_plate_tiny.msh")
model = GmshDiscreteModel(msh_file)

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = 1#μ[1]*(sin(π*t/tf)*x[1]+cos(π*t/tf)*x[2])
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

Uₘ = 1.5
H = 0.41
inflow(μ,t) = 1-cos(π*t/tf)+sin(π*t/μ[2]*tf)/μ[3]
g_in(x,μ,t) = VectorValue(4*Uₘ*x[2]*(H-x[2])/(H^2)*inflow(μ,t),0.0)#VectorValue(-x[2]*(H-x[2])*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_w(x,μ,t) = VectorValue(0.0,0.0)
g_w(μ,t) = x->g_w(x,μ,t)
gμt_w(μ,t) = TransientParamFunction(g_w,μ,t)
g_c(x,μ,t) = VectorValue(0.0,0.0)
g_c(μ,t) = x->g_c(x,μ,t)
gμt_c(μ,t) = TransientParamFunction(g_c,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

α = 1.e2
Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=degree)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + graddiv(u,v,dΩ)
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = (∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ)*(1/dt)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w,gμt_c])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)

solver_u = LUSolver()
solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6)

diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,test_p,test_p)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=true)
fesolver = ThetaMethod(solver,dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=1,nsnaps_test=1,nsnaps_mdeim=1)
fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)

r = get_realization(fesnaps)

r2 = r[2:2,:]
ur2,_ = fe_solutions(rbsolver,feop,xh0μ;r=r2)
