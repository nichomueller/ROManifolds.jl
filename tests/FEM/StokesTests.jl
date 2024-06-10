using LinearAlgebra
using Gridap
using Gridap.CellData
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ODEs
using Gridap.Helpers
using Test
using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamODEs

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)
μ = ParamDataStructures._get_params(r)[3]

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = 1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = 1-cos(2π*t/tf)+sin(μ[2]*2π*t/tf)/μ[1]
g(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,8])
trial_u = TransientTrialParamFESpace(test_u,gμt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_jac,trian_jac_t)

fesolver = ThetaMethod(LUSolver(),dt,θ)

sol = solve(fesolver,feop,r,xh0μ)
Base.iterate(sol)

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_form(t,(u,p),(v,q)) = ∫(_a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
_res(t,(u,p),(v,q)) = _form(t,(u,p),(v,q))
_jac(t,(u,p),(du,dp),(v,q)) = _form(t,(du,dp),(v,q))
_mass(t,(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
_jac_t(t,(u,p),(dut,dpt),(v,q)) = _mass(t,(dut,dpt),(v,q))

_trial_u = TransientTrialFESpace(test_u,_g)
_trial = TransientMultiFieldFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())
_x0 = interpolate_everywhere([u0(μ),p0(μ)],_trial(0.0))

_feop = TransientSemilinearFEOperator(_mass,_res,_jac,_jac_t,_trial,test)
_sol = solve(fesolver,_feop,t0,tf,_x0)
Base.iterate(_sol)

for ((rt,xh),(_t,_xh)) in zip(sol,_sol)
  uh,ph = xh
  uh1 = param_getindex(uh,3)
  ph1 = param_getindex(ph,3)
  _uh,_ph = _xh
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "failed at time $t"
  @check get_free_dof_values(ph1) ≈ get_free_dof_values(_ph) "failed at time $t"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end
