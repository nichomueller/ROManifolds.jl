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
dt = 0.01
t0 = 0.0
tf = 0.1

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

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

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
feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_jac,trian_jac_t)
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,induced_norm,ptspace,
  trial,test,trian_res,trian_jac)
feop = TransientParamLinNonlinFEOperator(feop_lin,feop_nlin)

fesolver = ThetaMethod(NewtonRaphsonSolver(LUSolver(),1e-10,20),dt,θ)

sol = solve(fesolver,feop,r,xh0μ)
Base.iterate(sol)

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_form(t,(u,p),(v,q)) = ∫(_a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
_res(t,(u,p),(v,q)) = _form(t,(u,p),(v,q)) + c(u,v,dΩ)
_jac(t,(u,p),(du,dp),(v,q)) = _form(t,(du,dp),(v,q)) + dc(u,du,v,dΩ)
_mass(t,(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
_jac_t(t,(u,p),(dut,dpt),(v,q)) = _mass(t,(dut,dpt),(v,q))

_trial_u = TransientTrialFESpace(test_u,_g)
_trial = TransientMultiFieldFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())
_x0 = interpolate_everywhere([u0(μ),p0(μ)],_trial(0.0))

_feop = TransientSemilinearFEOperator(_mass,_res,_jac,_jac_t,_trial,test)
_sol = solve(fesolver,_feop,t0,tf,_x0)
Base.iterate(_sol)

__form(t,(u,p),(v,q)) = ∫(_a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
__res(t,(u,p),(v,q)) = ∫(v⋅∂t(u))dΩ + _form(t,(u,p),(v,q)) + c(u,v,dΩ)
__jac(t,(u,p),(du,dp),(v,q)) = _form(t,(du,dp),(v,q)) + dc(u,du,v,dΩ)
__jac_t(t,(u,p),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
__feop = TransientFEOperator(__res,__jac,__jac_t,_trial,test)
__sol = solve(fesolver,__feop,t0,tf,_x0)
Base.iterate(__sol)

for ((t,xh),(_t,_xh)) in zip(_sol,__sol)
  uh,ph = xh
  _uh,_ph = _xh
  @check get_free_dof_values(uh) ≈ get_free_dof_values(_uh) "failed at time $t"
  @check get_free_dof_values(ph) ≈ get_free_dof_values(_ph) "failed at time $t"
end

for ((rt,xh),(_t,_xh)) in zip(sol,_sol)
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

jfeop = join_operators(feop)

nparams = 3
x = ParamArray([mortar([rand(num_free_dofs(test_u)),rand(num_free_dofs(test_p))]) for _ = 1:nparams])
r0 = FEM.get_at_time(r,:initial)
ff = FEFunction(trial(r0),x)
xh = TransientCellField(ff,(ff,))

du = get_trial_fe_basis(test)
dv = get_fe_basis(test)

_ff = FEM._getindex(ff,1)
_xh = TransientCellField(_ff,(_ff,))

function _check_equality(a::DomainContribution,b::DomainContribution)
  _first(y) = lazy_map(x->getindex(x,1),y)
  return lazy_map(x->getindex(x,1),_first(a[Ω])) ≈ _first(b[Ω])
end

DC = jfeop.op.jacs[1](get_params(r),0.0,xh,du,dv)
_DC = _feop.jacs[1](0.0,_xh,du,dv)
_check_equality(DC,_DC)

DC = jfeop.op.jacs[2](get_params(r),0.0,xh,du,dv)
_DC = _feop.jacs[2](0.0,_xh,du,dv)
DC[Ω][1][1] ≈ _DC[Ω][1][1]

DC = jfeop.op.res(get_params(r),0.0,xh,dv)
_DC = _feop.res(0.0,_xh,dv)
_check_equality(DC,_DC)

snaps = Snapshots(collect(sol.odesol),r)


# sol = sol.odesol do this once
r0 = FEM.get_at_time(sol.r,:initial)
cache = allocate_odecache(sol.solver,sol.odeop,r0,sol.us0)
state0,cache = ode_start(sol.solver,sol.odeop,r0,sol.us0,cache)
statef = copy.(state0)

w0 = state0[1]
odeslvrcache,odeopcache = cache
uθ,sysslvrcache = odeslvrcache

sysslvr = fesolver.sysslvr

x = statef[1]
dtθ = θ*dt
shift!(r0,dtθ)
function usx(x)
  copy!(uθ,w0)
  axpy!(dtθ,x,uθ)
  (uθ,x)
end
ws = (dtθ,1)

update_odeopcache!(odeopcache,sol.odeop,r0)

stageop = NonlinearParamStageOperator(sol.odeop,odeopcache,r0,usx,ws)

B = residual(stageop,x)
A = jacobian(stageop,x)

# residual
b = allocate_residual(stageop,x)
# residual!(b,stageop,x)
odeop,odeopcache = stageop.odeop,stageop.odeopcache
rx = stageop.rx
ussx = stageop.usx(x)
# residual!(b,odeop,rx,ussx,odeopcache)
uh = ODEs._make_uh_from_us(odeop,ussx,odeopcache.Us)
test = get_test(odeop.op)
v = get_fe_basis(test)
assem = get_assembler(odeop.op,rx)
fill!(b,zero(eltype(b)))
mydc = get_res(odeop.op)(get_params(rx),get_times(rx),uh,v)
vecdata = collect_cell_vector(test,mydc)
assemble_vector_add!(b,assem,vecdata)

#######################################
# _sol = _sol.odesltn do this once

t0 = 0.005

_cache = allocate_odecache(_sol.odeslvr,_sol.odeop,t0,_sol.us0)
_state0,_cache = ode_start(_sol.odeslvr,_sol.odeop,t0,_sol.us0,_cache)
_statef = copy.(_state0)

_w0 = _state0[1]
_odeslvrcache,_odeopcache = _cache
_uθ,_sysslvrcache = _odeslvrcache

_sysslvr = _sol.odeslvr.sysslvr

_x = _statef[1]
function _usx(x)
  copy!(_uθ,_w0)
  axpy!(dtθ,x,_uθ)
  (_uθ,x)
end

update_odeopcache!(_odeopcache,_sol.odeop,t0)

_stageop = NonlinearStageOperator(_sol.odeop,_odeopcache,t0,_usx,ws)

_B = residual(_stageop,_x)
_A = jacobian(_stageop,_x)

_b = allocate_residual(_stageop,_x)
_odeop,_odeopcache = _stageop.odeop,_stageop.odeopcache
_ussx = _stageop.usx(_x)
_uh = ODEs._make_uh_from_us(_sol.odeop,_ussx,_odeopcache.Us)
_test = get_test(_sol.odeop.tfeop)
_assem = get_assembler(_sol.odeop.tfeop)
fill!(_b,zero(eltype(_b)))
_mydc = get_res(_sol.odeop.tfeop)(t0,_uh,v)
_vecdata = collect_cell_vector(_test,_mydc)
assemble_vector_add!(_b,_assem,_vecdata)
