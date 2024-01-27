# module TransientParamFEOperatorsTests

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.CellData
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM

θ = 0.4

# Time stepping
t0 = 0.0
tf = 1.0
dt = 0.1

# transient parametric space
pranges = fill([1.,10.],3)
tdomain = dt:dt:tf
tpspace = TransientParamSpace(pranges,tdomain)
r = realization(tpspace;nparams=2)
μ,t = get_params(r),get_times(r)

# Analytical functions
sol(x,μ,t) = (1.0-x[1])*x[1]*(1.0-x[2])*x[2]*(t+3.0)*sum(μ)
sol(μ,t) = x -> sol(x,μ,t)
u(μ,t) = 𝑓ₚₜ(sol,μ,t)
f(μ,t) = x -> ∂t(u(μ,t))(x)-Δ(u(μ,t))(x)
∂tu(μ,t) = ∂t(u(μ,t))

# Domain and triangulations
domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = FESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
U = TransientTrialParamFESpace(V0,u)
Ω = Triangulation(model)
# Γ = BoundaryTriangulation(model,tags="boundary")
degree = 2*order
dΩ = Measure(Ω,degree)
# dΓ = Measure(Γ,degree)
# nΓ = get_normal_vector(Γ)
# h = 1/partition[1]

# Affine FE operator
a(u,v) = ∫(∇(v)⊙∇(u))dΩ #- ∫(0.0*v⋅(nΓ⋅∇(u))  + u⋅(nΓ⋅∇(v)) - 10/h*(v⋅u))dΓ
m(u,v) = ∫(v*u)dΩ
b(μ,t,v) = ∫(v*f(μ,t))dΩ #- ∫(u(t)⋅(nΓ⋅∇(v)) - 10/h*(v⋅u(t)) )dΓ
res(μ,t,u,v) = a(u,v) + m(∂t(u),v) - b(μ,t,v)
lhs(μ,t,u,v) = m(∂t(u),v)
rhs(μ,t,u,v) = b(μ,t,v) - a(u,v)
irhs(μ,t,u,v) = b(μ,t,v) - a(u,v)#∫( -1.0*(∇(v)⊙∇(u)))dΩ
erhs(μ,t,u,v) = ∫( 0.0*(∇(v)⊙∇(u)))dΩ#b(v,t)
jac(μ,t,u,du,v) = a(du,v)
jac_t(μ,t,u,dut,v) = m(dut,v)
op = TransientParamFEOperator(res,jac,jac_t,tpspace,U,V0)
# opRK = TransientRungeKuttaFEOperator(lhs,rhs,jac,jac_t,U,V0)
# opIMEXRK = TransientIMEXRungeKuttaFEOperator(lhs,irhs,erhs,jac,jac_t,U,V0)

# Initial solution
uh0(μ) = interpolate_everywhere(u(μ,0.0),U(μ,0.0))
∂tuh0(μ) = interpolate_everywhere(∂tu(μ,0.0),U(μ,0.0))

function test_ode_solver(ode_solver,op,xh0)
  sol_t = solve(ode_solver,op,xh0;nparams=2)

  l2(w) = w*w

  tol = 1.0e-6
  _t_n = t0

  for (uh_tn, tn) in sol_t
    # global _t_n
    _t_n += dt
    @test tn≈_t_n
    e = u(μ,tn) - uh_tn
    el2 = sqrt(sum( ∫(l2(e))dΩ ))
    @test all(el2 .< tol)
  end

  @test length( [uht for uht in sol_t] ) == ceil((tf - t0)/dt)

end

# Linear solver
ls = LUSolver()

# ODE solvers
ode_solvers = []
push!(ode_solvers,(ThetaMethod(ls,dt,θ),op,uh0))
push!(ode_solvers,(BackwardEuler(ls,dt),op,uh0))
push!(ode_solvers,(MidPoint(ls,dt),op,uh0))
# push!(ode_solvers,(GeneralizedAlpha(ls,dt,1.0),op,(uh0,∂tuh0)))
# push!(ode_solvers,(RungeKutta(ls,ls,dt,:BE_1_0_1),opRK,uh0))
# push!(ode_solvers,(RungeKutta(ls,ls,dt,:CN_2_0_2),opRK,uh0))
# push!(ode_solvers,(RungeKutta(ls,ls,dt,:SDIRK_2_0_2),opRK,uh0))
# push!(ode_solvers,(IMEXRungeKutta(ls,ls,dt,:IMEX_FE_BE_2_0_1),opIMEXRK,uh0))
for ode_solver in ode_solvers
  test_ode_solver(ode_solver...)
end
#

r = realization(tpspace;nparams=2)
μ = get_params(r)
u0 = get_free_dof_values(uh0(μ))
uf = get_free_dof_values(uh0(μ))

odeop = get_algebraic_operator(op)

ode_cache = allocate_cache(odeop,r)
vθ = similar(u0)
nl_cache = nothing

# tf = t0+dt

# Nonlinear ThetaMethod
ode_solver = ThetaMethod(ls,dt,θ)
ode_solver.θ == 0.0 ? dtθ = dt : dtθ = dt*ode_solver.θ
rθ = get_at_time(r,:initial)
shift_time!(rθ,dtθ)
ode_cache = update_cache!(ode_cache,odeop,rθ)

using Gridap.ODEs.ODETools: ThetaMethodNonlinearOperator
nlop = ThetaMethodParamOperator(odeop,rθ,dtθ,u0,ode_cache,vθ)

nl_cache = solve!(uf,ode_solver.nls,nlop,nl_cache)

K = nl_cache.A
h = nl_cache.b

# Steady version of the problem to extract the Laplacian and mass matrices
# tf = 0.1
shift_time!(rθ,dt*(1-θ))
Utf = U(rθ)
# fst(x) = -Δ(u(tf))(x)
fθ(x) = f(get_params(rθ),get_times(rθ))(x)

function extract_matrix_vector(a,fst)
  btf(v) = ∫(v*fst)dΩ
  op = AffineFEOperator(a,btf,Utf,V0)
  ls = LUSolver()
  solver = LinearFESolver(ls)
  uh = solve(solver,op)

  tol = 1.0e-6
  e = uh-u(tf)
  l2(e) = e*e
  l2e = sqrt(sum( ∫(l2(e))dΩ ))
  # @test l2e < tol

  Ast = op.op.matrix
  bst = op.op.vector

  @test uh.free_values ≈ Ast \ bst

  return Ast, bst
end

A,vec = extract_matrix_vector(a,fst)

gst(x) = u(tf)(x)
m(u,v) = ∫(u*v)dΩ

M,_ = extract_matrix_vector(m,gst)

@test vec ≈ h
@test A+M/(θ*dt) ≈ K

rhs
h


# end #module


_b(x,μ,t) = sum(μ)*sum(x)*t
_b(μ,t) = x -> _b(x,μ,t)
bb = 𝑓ₚₜ(_b,μ,t)
dbb = ∂t(bb)

bb(x)
dbb(x)

_c(x,μ,t) = sum(μ)*sum(x)*t
_c(μ,t) = x -> _c(x,μ,t)
# cc(μ,t) = 𝑓ₚₜ(_c(μ,t),μ,t)
cc(μ,t) = 𝑓ₚₜ(_c,μ,t)
dcc(μ,t) = ∂t(cc(μ,t))


_U = TransientTrialParamFESpace(V0,cc)
interpolate_everywhere(cc(μ,0.0),_U(μ,0.0))
interpolate_everywhere(dcc(μ,0.0),_U(μ,0.0))
