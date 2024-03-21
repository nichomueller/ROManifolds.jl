using LinearAlgebra
using Gridap
using Gridap.CellData
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ODEs
using Gridap.Helpers
using Test
using Mabla.FEM

θ = 0.5
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)
μ = FEM._get_params(r)[3]

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
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

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialParamFESpace(test,gμt)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

########################## GRIDAP DEFS ##############################

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_f(x,t) = f(x,μ,t)
_f(t) = x->_f(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_h(x,t) = h(x,μ,t)
_h(t) = x->_h(x,t)

_trial = TransientTrialFESpace(test,_g)
_u0 = interpolate_everywhere(x->0.0,_trial(0.0))

########################## LINEAR ############################

fesolver = ThetaMethod(LUSolver(),dt,θ)

mass(μ,t,dut,v) = ∫(v*dut)dΩ
stiffness(μ,t,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
rhs(μ,t,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn

feop = TransientParamLinearFEOperator((stiffness,mass),rhs,induced_norm,ptspace,trial,test)
sol = solve(fesolver,feop,r,uh0μ)
Base.iterate(sol)

_stiffness(t,du,v) = ∫(_a(t)*∇(v)⋅∇(du))dΩ
_mass(t,dut,v) = ∫(v*dut)dΩ
_rhs(t,v) = (-1)*(∫(_f(t)*v)dΩ + ∫(_h(t)*v)dΓn)

_feop = TransientLinearFEOperator((_stiffness,_mass),_rhs,_trial,test)
_sol = solve(fesolver,_feop,t0,tf,_u0)
Base.iterate(_sol)

for ((rt,uh),(_t,_uh)) in zip(sol,_sol)
  uh1 = FEM._getindex(uh,3)
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "$(get_free_dof_values(uh1)) != $(get_free_dof_values(_uh))"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end

########################## SEMILINEAR ############################

fesolver = ThetaMethod(NewtonRaphsonSolver(LUSolver(),1e-10,20),dt,θ)

mass(μ,t,dut,v) = ∫(v*dut)dΩ
res(μ,t,u,v) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn

feop = TransientParamSemilinearFEOperator(mass,res,induced_norm,ptspace,trial,test)
sol = solve(fesolver,feop,r,uh0μ)
Base.iterate(sol)

_mass(t,dut,v) = ∫(v*dut)dΩ
_res(t,u,v) = ∫(_a(t)*∇(v)⋅∇(u))dΩ - ∫(_f(t)*v)dΩ - ∫(_h(t)*v)dΓn

_feop = TransientSemilinearFEOperator(_mass,_res,_trial,test)
_sol = solve(fesolver,_feop,t0,tf,_u0)
Base.iterate(_sol)

for ((rt,uh),(_t,_uh)) in zip(sol,_sol)
  uh1 = FEM._getindex(uh,3)
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "$(get_free_dof_values(uh1)) != $(get_free_dof_values(_uh))"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end
