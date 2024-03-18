########################## SETTING #############################

using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField

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

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

########################## STOKES ############################

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0.0)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
jac(μ,t,u,(du,dp),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ + ∫(q*(∇⋅(du)))dΩ
jac_t(μ,t,u,(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial_u = TransientTrialParamFESpace(test_u,gμt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test,coupling)
feop = FEOperatorWithTrian(_feop,trian_res,trian_jac,trian_jac_t)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

sol = solve(fesolver,feop,xh0μ,r)
iterate(sol)

# gridap

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_f(x,t) = VectorValue(0.0,0.0)
_f(t) = x->_f(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_b(t,(v,q)) = ∫(_f(t)⋅v)dΩ
_a(t,(du,dp),(v,q)) = ∫(_a(t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
_m(t,(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ

_trial_u = TransientTrialFESpace(test_u,_g)
_trial = TransientMultiFieldFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())
_feop = TransientAffineFEOperator(_m,_a,_b,_trial,test)
_x0 = interpolate_everywhere([u0(μ),p0(μ)],_trial(0.0))

_sol = solve(fesolver,_feop,_x0,t0,tf)
iterate(_sol)

for ((xh,rt),(_xh,_t)) in zip(sol,_sol)
  uh,ph = xh
  uh1 = FEM._getindex(uh,3)
  ph1 = FEM._getindex(ph,3)
  _uh,_ph = _xh
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "failed at time $t"
  @check get_free_dof_values(ph1) ≈ get_free_dof_values(_ph) "failed at time $t"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end


########################## NAVIER-STOKES ############################

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

_feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,induced_norm,ptspace,trial,test)
feop_nlin = FEOperatorWithTrian(_feop_nlin,trian_res,trian_jac)
feop_lin_nlin = TransientParamLinearNonlinearFEOperator(feop,feop_nlin)

sol = solve(fesolver,feop_lin_nlin,xh0μ,r)
iterate(sol)

# gridap

_form(t,(u,p),(v,q)) = ∫(_a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
_res(t,(u,p),(v,q)) = ∫(v⋅∂t(u))dΩ + _form(t,(u,p),(v,q)) + c(u,v,dΩ)
_jac(t,(u,p),(du,dp),(v,q)) = _form(t,(du,dp),(v,q)) + dc(u,du,v,dΩ)
_jac_t(t,(u,p),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ

_trial_u = TransientTrialFESpace(test_u,_g)
_trial = TransientMultiFieldFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())
_feop = TransientFEOperator(_res,_jac,_jac_t,_trial,test)
_x0 = interpolate_everywhere([u0(μ),p0(μ)],_trial(0.0))

_sol = solve(fesolver,_feop,_x0,t0,tf)
iterate(_sol)

for ((xh,rt),(_xh,_t)) in zip(sol,_sol)
  uh,ph = xh
  uh1 = FEM._getindex(uh,3)
  ph1 = FEM._getindex(ph,3)
  _uh,_ph = _xh
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "failed at time $t"
  @check get_free_dof_values(ph1) ≈ get_free_dof_values(_ph) "failed at time $t"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end
