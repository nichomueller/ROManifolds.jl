using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM

θ = 0.2
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
μt = realization(ptspace,nparams=3)
μ = get_params(μt)

u(x,μ,t) = (1.0-x[1])*x[1]*(1.0-x[2])*x[2]*t*sum(μ)
u(μ,t) = x -> u(x,μ,t)
uμt(μ,t) = 𝑓ₚₜ(u,μ,t)
f(μ,t) = x -> ∂t(uμt)(x,μ,t)-Δ(uμt(μ,t))(x)
fμt(μ,t) = 𝑓ₚₜ(f,μ,t)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 2

reffe = ReferenceFE(lagrangian,Float64,order)
V0 = FESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
U = TransientTrialParamFESpace(V0,uμt)

Ω = Triangulation(model)
degree = 2*order
dΩ = Measure(Ω,degree)

a(u,v) = ∫(∇(v)⋅∇(u))dΩ
b(μ,t,v) = ∫(v*fμt(μ,t))dΩ

res(μ,t,u,v) = a(u,v) + ∫(∂t(u)*v)dΩ - b(μ,t,v)
jac(μ,t,u,du,v) = a(du,v)
jac_t(μ,t,u,dut,v) = ∫(dut*v)dΩ

op = TransientParamFEOperator(res,jac,jac_t,ptspace,U,V0)

uh0(μ) = interpolate_everywhere(uμt(μ,t0),U(μ,t0))

ls = LUSolver()
using Gridap.Algebra: NewtonRaphsonSolver
tol = 1.0
maxiters = 20
nls = NewtonRaphsonSolver(ls,tol,maxiters)
ode_solver = ThetaMethod(ls,dt,θ)

sol_t = solve(ode_solver,op,uh0,t0,tf)

l2(w) = w*w

tol = 1.0e-6
_t_n = t0

for (uh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  e = u(tn) - uh_tn
  el2 = sqrt(sum( ∫(l2(e))dΩ ))
  @test el2 < tol
end

#############################
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools

𝒯 = CartesianDiscreteModel((0,1,0,1),(20,20))
Ω = Interior(𝒯)
dΩ = Measure(Ω,2)
T = Float64
reffe_u = ReferenceFE(lagrangian,T,2)
reffe_p = ReferenceFE(lagrangian,T,1)
g(x,t::Real) = 0.0
g(t::Real) = x -> g(x,t)
mfs = BlockMultiFieldStyle()
test_u = TestFESpace(𝒯,reffe_u;conformity=:H1,dirichlet_tags="boundary")
trial_u = TransientTrialFESpace(test_u,g)
test_p = TestFESpace(𝒯,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
Yb  = TransientMultiFieldFESpace([test_u,test_p];style=mfs)
Xb  = TransientMultiFieldFESpace([trial_u,trial_p];style=mfs)
κ(t) = 1.0 + 0.95*sin(2π*t)
f(t) = sin(π*t)
res(t,(u,p),(v,q)) = ∫( ∂t(u)*v + κ(t)*(∇(v)⊙∇(u)) - p*(∇⋅(v)) - q*(∇⋅(u)) - f(t)*v )dΩ
jac(t,(u,p),(du,dp),(v,q)) = ∫( κ(t)*(∇(du)⋅∇(v)) - dp*(∇⋅(v)) - q*(∇⋅(du)) )dΩ
jac_t(t,(u,p),(duₜ,dpₜ),(v,q)) = ∫( duₜ*v )dΩ
op = TransientFEOperator(res,jac,jac_t,U,V)
m(t,u,v) = ∫( u*v )dΩ
a(t,u,v) = ∫( κ(t)*(∇(u)⋅∇(v)) )dΩ
b(t,v) = ∫( f(t)*v )dΩ
op_Af = TransientAffineFEOperator(m,a,b,U,V)
linear_solver = LUSolver()
Δt = 0.1
θ = 0.5
ode_solver = ThetaMethod(linear_solver,Δt,θ)
u₀ = interpolate_everywhere(0.0,U(0.0))
t₀ = 0.0
T = 10.0
uₕₜ = solve(ode_solver,op,u₀,t₀,T)

g0(x) = 0.0
trial_u = TrialFESpace(test_u,g0)
trial_p = TrialFESpace(test_p)
Yb  = MultiFieldFESpace([test_u,test_p];style=mfs)
Xb  = MultiFieldFESpace([trial_u,trial_p];style=mfs)
biform((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 - u1⋅v2)*dΩ
liform((v1,v2)) = ∫(v1 - v2)*dΩ
ub = get_trial_fe_basis(Xb)
vb = get_fe_basis(Yb)
bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
bvecdata = collect_cell_vector(Yb,liform(vb))
