using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed

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
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 20
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"inlet",[7])
add_tag_from_tags!(labels,"dirichlet0",[1,2,3,4,5,6])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]*exp(sin(π*t/tf)*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = 1-cos(π*t/tf)+sin(π*t/(μ[2]*tf))/μ[3]
g_in(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
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

α = 1.e6
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
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)

solver_u = LUSolver()
solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)

diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,test_p,test_p)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=true)
fesolver = ThetaMethod(solver,dt,θ)

ϵ = 1e-5
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)

#=  =#using Gridap.ODEs

r = realization(feop)
r0 = ParamDataStructures.get_at_time(r,:initial)
r = r0
odeop = get_algebraic_operator(feop.op.op)
solver = fesolver
us0 = (get_free_dof_values(xh0μ(r0.params)),)
odecache = allocate_odecache(solver,odeop,r0,us0)
state0,cache = ode_start(solver,odeop,r0,us0,odecache)
statef = copy.(state0)

odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache

sysslvr = solver.sysslvr
x = statef[1]
fill!(x,zero(eltype(x)))
dtθ = θ*dt
shift!(r,dtθ)
usx = (state0[1],x)
ws = (dtθ,1)
update_odeopcache!(odeopcache,odeop,r)
stageop = LinearParamStageOperator(odeop,odeopcache,r,usx,ws,A,b,reuse,sysslvrcache)

# sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)
A = stageop.A
ss = symbolic_setup(sysslvr,A)
ns = numerical_setup(ss,A)

#
# numerical_setup(ss,A)
# Pr_ns = numerical_setup(symbolic_setup(ss.solver.Pr,A),A)
ss = symbolic_setup(ss.solver.Pr,A)
block_ns    = map(numerical_setup,ss.block_ss,diag(ss.block_caches))
y = mortar(map(allocate_in_domain,diag(ss.block_caches))); fill!(y,0.0)
w = allocate_in_range(A); fill!(w,0.0)
work_caches = w,y

b = stageop.b
rmul!(b,-1)
# solve!(x,ns,b)
solver, A, Pl, Pr, caches = ns.solver, ns.A, ns.Pl_ns, ns.Pr_ns, ns.caches
V, Z, zl, H, g, c, s = caches
m   = GridapSolvers.LinearSolvers.krylov_cache_length(ns)
log = solver.log

fill!(V[1],zero(eltype(V[1])))
fill!(zl,zero(eltype(zl)))

# Initial residual
GridapSolvers.LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
β    = norm(V[1])
done = GridapSolvers.LinearSolvers.init!(log,β)
# Arnoldi process
j = 1
V[1] ./= β
fill!(H,0.0)
fill!(g,0.0); g[1] = β

# Expand Krylov basis if needed
if j > m
  H, g, c, s = GridapSolvers.LinearSolvers.expand_krylov_caches!(ns)
  m = GridapSolvers.LinearSolvers.krylov_cache_length(ns)
end

# Arnoldi orthogonalization by Modified Gram-Schmidt
fill!(V[j+1],zero(eltype(V[j+1])))
fill!(Z[j],zero(eltype(Z[j])))
# GridapSolvers.LinearSolvers.krylov_mul!(V[j+1],A,V[j],Pr,Pl,Z[j],zl)
wr = Z[j]
x = V[j]
# solve!(wr,Pr,x)
NB = length(Pr.block_ns)
c = Pr.solver.coeffs
w,y = Pr.work_caches
mats = Pr.block_caches
iB = NB

wi  = w[Block(iB)]
copy!(wi,b[Block(iB)])
for jB in iB+1:NB
  cij = c[iB,jB]
  if abs(cij) > eps(cij)
    xj = x[Block(jB)]
    mul!(wi,mats[iB,jB],xj,-cij,1.0)
  end
end

# Solve diagonal block
nsi = Pr.block_ns[iB]
xi  = x[Block(iB)]
yi  = y[Block(iB)]
solve!(yi,nsi,wi)
copy!(xi,yi) # Remove this with PA 0.4
