# module TransientPFETests

using Gridap
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Test
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM

θ = 1.0
dt = 0.1

pranges = fill([0,1],3)
tdomain = 0:dt:1
ptspace = TransientParamSpace(pranges,tdomain)
μt = realization(ptspace,nparams=3)
μt0 = FEM.get_at_time(μt,:initial)
μtf = FEM.get_at_time(μt,:final)
μ = get_parameters(μt)

u(x,μ,t) = (x[1] + x[2])*t*sum(μ)
u(μ,t) = x -> u(x,μ,t)
uμt(μ,t) = 𝑓ₚₜ(u,μ,t)
∇u(x,μ,t) = VectorValue(1,1)*t*sum(μ)
∇u(μ,t) = x -> ∇u(x,μ,t)
∇uμt(μ,t) = 𝑓ₚₜ(∇u,μ,t)
import Gridap: ∇

∂tu(μ,t) = x -> (x[1]+x[2])*sum(μ)
∂tuμt(μ,t) = 𝑓ₚₜ(∂tu,μ,t)
import Gridap.ODEs.TransientFETools: ∂t
∂t(::typeof(uμt)) = ∂tuμt
@test ∂t(uμt) === ∂tuμt

f(μ,t) = x -> (x[1]+x[2])*sum(μ)
fμt(μ,t) = 𝑓ₚₜ(f,μ,t)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = TestFESpace(
  model,
  reffe,
  conformity=:H1,
  dirichlet_tags="boundary"
)

U = TransientTrialParamFESpace(V0,uμt)
@test test_transient_trial_fe_space(U,μ)

U0 = U(μ,1.0)
ud0 = copy(get_dirichlet_dof_values(U0))
_ud0 = get_dirichlet_dof_values(U0)
U1 = U(μ,2.0)
ud1 = copy(get_dirichlet_dof_values(U1))
_ud1 = get_dirichlet_dof_values(U1)
@test all(ud0 ≈ 0.5ud1)

Ut = ∂t(U)
Ut.dirichlet_pt
Ut0 = Ut(μ,0.0)
Ut0.dirichlet_values

Ut1 = Ut(μ,1.0)
utd0 = copy(get_dirichlet_dof_values(Ut0))
utd1 = copy(get_dirichlet_dof_values(Ut1))
@test all(utd0 == utd1)
@test all(utd1 == ud0)

Ω = Triangulation(model)
degree = 2
dΩ = Measure(Ω,degree)

a(u,v) = ∫(∇(v)⋅∇(u))dΩ
b(v,μ,t) = ∫(v*f(μ,t))dΩ
bμt(v,μ,t) = ∫(v*fμt(μ,t))dΩ

res(μ,t,u,v) = a(u,v) + ∫(∂t(u)*v)dΩ - bμt(v,μ,t)
jac(μ,t,u,du,v) = a(du,v)
jac_t(μ,t,u,dut,v) = ∫(dut*v)dΩ

using Gridap.FESpaces: allocate_residual, allocate_jacobian, residual!, jacobian!

op = TransientParamFEOperator(res,jac,jac_t,ptspace,U,V0)
odeop = get_algebraic_operator(op)
cache = allocate_cache(odeop,μt)

U0 = U(μt0)
u0(x,μ) = u(x,μ,0.0)
u0(μ) = x->u0(x,μ)
u0μ = 𝑓ₚ(u0,μ)

uh = interpolate_everywhere(u0μ,U0)
r = allocate_residual(op,μt0,uh,cache)
J = allocate_jacobian(op,μt0,uh,cache)
uh0 = interpolate_everywhere(u0μ,U0)
xh = TransientCellField(uh,(uh0,))
residual!(r,op,μt0,xh,cache)
jacobian!(J,op,μt0,xh,1,1.0,cache)
jacobian!(J,op,μt0,xh,2,10.0,cache)

map(μ,U0,uh,r,J) do μ,U0,uh,r,J
  _res(u,v) = a(u,v) + 10.0*∫(u*v)dΩ - b(v,μ,0.0)
  _jac(u,du,v) = a(du,v) + 10.0*∫(du*v)dΩ
  _op = FEOperator(_res,_jac,U0,V0)

  _r = allocate_residual(_op,uh)
  _J = allocate_jacobian(_op,uh)
  residual!(_r,_op,uh)
  jacobian!(_J,_op,uh)
  @test all(r.≈_r)
  @test all(J.≈_J)
end

U0 = U(μt0)
uh0 = interpolate_everywhere(u0μ,U0)
@test test_transient_fe_operator(op,uh0,μt0)

ls = LUSolver()
# using LineSearches: BackTracking
tol = 1.0
maxiters = 20
using Gridap.Algebra: NewtonRaphsonSolver
nls = NewtonRaphsonSolver(ls,tol,maxiters)
ode_solver = ThetaMethod(nls,dt,1.0)
ic(μ) = interpolate_everywhere(uμt(μ,0.0),U(μ,0.0))
@test test_transient_fe_solver(ode_solver,op,ic,μt0)

xh = TransientCellField(uh,(uh,))
residual!(r,op,μtf,xh,cache)
jacobian!(J,op,μtf,xh,1,1.0,cache)
jacobian!(J,op,μtf,xh,2,10.0,cache)

v0 = get_free_dof_values(uh0)
solver = ode_solver
ode_cache = allocate_cache(odeop,μtf)
cache = nothing
uf = copy(v0)
update_cache!(ode_cache,odeop,μtf)
vf = copy(v0)
nlop = ThetaMethodParamOperator(odeop,μtf,dt,v0,ode_cache,vf)

x = copy(nlop.u0)

b1 = allocate_residual(nlop,x)
residual!(b1,nlop,x)
b2 = allocate_residual(nlop,x)
residual!(b2,nlop.odeop,nlop.r,(x,10.0*x),nlop.ode_cache)
@test all(b1 ≈ b2)
J1 = allocate_jacobian(nlop,x)
jacobian!(J1,nlop,x)
J2 = allocate_jacobian(nlop,x)
jacobian!(J2,nlop.odeop,nlop.r,(x,10.0*x),1,1.0,nlop.ode_cache)
jacobian!(J2,nlop.odeop,nlop.r,(x,10.0*x),2,10.0,nlop.ode_cache)
@test all(J1 ≈ J2)
using Gridap.Algebra: test_nonlinear_operator
test_nonlinear_operator(nlop,x,b1,jac=J1)

x .= 0.0
r = allocate_residual(nlop,x)
residual!(r,nlop,x)
J = allocate_jacobian(nlop,x)
jacobian!(J,nlop,x)

cache = solve!(uf,solver.nls,nlop)
dx = cache.dx
ns = cache.ns

function linsolve!(x,A,b)
  numerical_setup!(ns,A)
  solve!(x,ns,b)
end

p = copy(x)
p .= 0.0
l_sol = linsolve!(p,J,-r)
J*l_sol ≈ -r
x = x + l_sol
residual!(r,nlop,x)
map(r) do r
  @test all(abs.(r) .< 1e-6)
end

residual!(r,nlop,x)
jacobian!(J,nlop,x)
p .= 0.0
l_sol = linsolve!(p,J,-r)

cache = solve!(uf,solver.nls,nlop)
@test all(uf ≈ x)

using Gridap.FESpaces: get_algebraic_operator
odeop = get_algebraic_operator(op)
sol_ode_t = solve(ode_solver,odeop,ω,μt)

test_ode_solution(sol_ode_t)
_t_n = t0
for (u_n, t_n) in sol_ode_t
  global _t_n
  _t_n += dt
  @test t_n≈_t_n
  @test all(u_n .≈ t_n)
end

ode_solver = ThetaMethod(nls,dt,θ)
sol_ode_t = solve(ode_solver,odeop,u0,t0,tF)
test_ode_solution(sol_ode_t)
_t_n = t0
un, tn = Base.iterate(sol_ode_t)
for (u_n, t_n) in sol_ode_t
  global _t_n
  _t_n += dt
  @test t_n≈_t_n
  @test all(u_n .≈ t_n)
end

sol_t = solve(ode_solver,op,uh0,t0,tF)
@test test_transient_fe_solution(sol_t)

_t_n = 0.0
for (u_n, t_n) in sol_t
  global _t_n
  _t_n += dt
  @test t_n≈_t_n
  @test all(u_n.free_values .≈ t_n)
end

l2(w) = w*w

# h1(w) = a(w,w) + l2(w)

_t_n = t0
for (uh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  @test tn≈_t_n
  e = u(tn) - uh_tn
  el2 = sqrt(sum( ∫(l2(e))dΩ ))
  @test el2 < tol
  # writevtk(trian,"sol at time: $tn",cellfields=["u" => uh_tn])
end

# end #module
