using Test
using Gridap.Arrays
using Gridap.TensorValues
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.Algebra
using Gridap.ODEs
using Gridap.Helpers
using SparseArrays
using SparseMatricesCSR
using Gridap.FESpaces
using Gridap.CellData
using Gridap.Algebra
using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamODEs

θ = 0.5
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)
μ = ParamDataStructures._get_params(r)[3]

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

w0(x,μ) = 0
w0(μ) = x->w0(x,μ)
w0μ(μ) = ParamFunction(w0,μ)

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialParamFESpace(test,gμt)
uh0μ(μ) = interpolate_everywhere(w0μ(μ),trial(μ,t0))

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

stiffness(μ,t,u,v) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v) = ∫(v*uₜ)dΩ
rhs(μ,t,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v) = mass(μ,t,∂t(u),v) + stiffness(μ,t,u,v) - rhs(μ,t,v)

feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,trial,test)
sol = solve(fesolver,feop,r,uh0μ)
Base.iterate(sol)

_stiffness(t,u,v) = ∫(_a(t)*∇(v)⋅∇(u))dΩ
_mass(t,uₜ,v) = ∫(v*uₜ)dΩ
_rhs(t,v) = (-1)*(∫(_f(t)*v)dΩ + ∫(_h(t)*v)dΓn)

_feop = TransientLinearFEOperator((_stiffness,_mass),_rhs,_trial,test)
_sol = solve(fesolver,_feop,t0,tf,_u0)
Base.iterate(_sol)

for ((rt,uh),(_t,_uh)) in zip(sol,_sol)
  uh1 = param_getindex(uh,3)
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "$(get_free_dof_values(uh1)) != $(get_free_dof_values(_uh))"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values "$(uh1.dirichlet_values) != $(_uh.dirichlet_values)"
end

########################## SEMILINEAR ############################

function Algebra._check_convergence(nls,b,m0)
  m = maximum(abs,b)
  println(m)
  return m < nls.tol * m0
end

fesolver = ThetaMethod(NewtonRaphsonSolver(LUSolver(),1e-10,20),dt,θ)

mass(μ,t,uₜ,v) = ∫(v*uₜ)dΩ
res(μ,t,u,v) = mass(μ,t,∂t(u),v) + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn

feop = TransientParamSemilinearFEOperator(mass,res,induced_norm,ptspace,trial,test)
sol = solve(fesolver,feop,r,uh0μ)
Base.iterate(sol)

_mass(t,uₜ,v) = ∫(v*uₜ)dΩ
_res(t,u,v) = ∫(_a(t)*∇(v)⋅∇(u))dΩ - ∫(_f(t)*v)dΩ - ∫(_h(t)*v)dΓn

_feop = TransientSemilinearFEOperator(_mass,_res,_trial,test)
_sol = solve(fesolver,_feop,t0,tf,_u0)
Base.iterate(_sol)

for ((rt,uh),(_t,_uh)) in zip(sol,_sol)
  uh1 = FEM.param_getindex(uh,3)
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "$(get_free_dof_values(uh1)) != $(get_free_dof_values(_uh))"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end

########################## LINEAR-TRIAN ############################

fesolver = ThetaMethod(LUSolver(),dt,θ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
sol = solve(fesolver,feop,r,uh0μ)
Base.iterate(sol)

_stiffness(t,u,v) = ∫(_a(t)*∇(v)⋅∇(u))dΩ
_mass(t,uₜ,v) = ∫(v*uₜ)dΩ
_rhs(t,v) = (-1)*(∫(_f(t)*v)dΩ + ∫(_h(t)*v)dΓn)

_feop = TransientLinearFEOperator((_stiffness,_mass),_rhs,_trial,test)
_sol = solve(fesolver,_feop,t0,tf,_u0)
Base.iterate(_sol)

for ((rt,uh),(_t,_uh)) in zip(sol,_sol)
  uh1 = ParamDataStructures.param_getindex(uh,3)
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "$(get_free_dof_values(uh1)) != $(get_free_dof_values(_uh))"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end

########################## SEMILINEAR-TRIAN ############################

fesolver = ThetaMethod(NewtonRaphsonSolver(LUSolver(),1e-10,20),dt,θ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_jac = (Ω,Γn)
trian_mass = (Ω,)

feop = TransientParamSemilinearFEOperator(mass,res,induced_norm,ptspace,trial,test,
  trian_res,trian_jac,trian_mass)
sol = solve(fesolver,feop,r,uh0μ)
Base.iterate(sol)

_mass(t,uₜ,v) = ∫(v*uₜ)dΩ
_res(t,u,v) = ∫(_a(t)*∇(v)⋅∇(u))dΩ - ∫(_f(t)*v)dΩ - ∫(_h(t)*v)dΓn

_feop = TransientSemilinearFEOperator(_mass,_res,_trial,test)
_sol = solve(fesolver,_feop,t0,tf,_u0)
Base.iterate(_sol)

for ((rt,uh),(_t,_uh)) in zip(sol,_sol)
  uh1 = ParamDataStructures.param_getindex(uh,3)
  t = get_times(rt)
  @check t ≈ _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "$(get_free_dof_values(uh1)) != $(get_free_dof_values(_uh))"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end




#
sol = sol.odesol
r0 = ParamDataStructures.get_at_time(sol.r,:initial)
cache = allocate_odecache(sol.solver,sol.odeop,r0,sol.us0)
state0,cache = ode_start(sol.solver,sol.odeop,r0,sol.us0,cache)
statef = copy.(state0)
u0 = state0[1]
odecache = cache
odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache

sysslvr = fesolver.sysslvr

x = statef[1]
fill!(x,zero(eltype(x)))
dtθ = θ*dt
shift!(r,dtθ)
usx = (u0,x)
ws = (dtθ,1)

odeop = sol.odeop

update_odeopcache!(odeopcache,odeop,r0)

# LinearParamStageOperator(odeop,odeopcache,r0,usx,ws,A,b,reuse,sysslvrcache)
# A = jacobian!(A,odeop,r0,usx,ws,odeopcache)
using LinearAlgebra
LinearAlgebra.fillstored!(A,zero(eltype(A)))
# jacobian_add!(A,odeop,r0,usx,ws,odeopcache)

jacs = get_jacs(odeop.op)
uh = ODEs._make_uh_from_us(odeop,usx,odeopcache.Us)
μ,t = r0.params,0.0
du,v = get_trial_fe_basis(test),get_fe_basis(test)
dc = DomainContribution()
dc = dc + ws[1]*jacs[1](μ,t,uh,du,v)
# dc = dc + ws[2]*jacs[2](μ,t,uh,du,v)
aa,bb = dc,ws[1]*jacs[1](μ,t,uh,du,v)
c = copy(aa)
c.dict[Ω] = bb[Ω]

B = feop.res(μ,1.0,uh,v)

f1 = ws[1]*jacs[1](μ,t,uh,du,v)
f2 = jacs[1](μ,t,uh,du,v)
f3 = ws[1]*f2

d = DomainContribution()

using FillArrays
array_old = f2[Ω]
s = size(get_cell_map(Ω))
array_new = lazy_map(Broadcasting(*),Fill(ws[1],s),array_old)

cache = return_cache(Broadcasting(*),ws[1],array_old[1])
pA = ParamDataStructures._to_param_quantities(ws[1],array_old[1])

#
μ = ParamRealization([rand(3),rand(3)])
t = [0.1,0.2]

odeop = get_algebraic_operator(feop)
urand = rand(num_free_dofs(test))
us = (ParamArray([urand for _ = 1:4]),)
Us = (trial(μ,t),)
uh = TransientCellField(EvaluationFunction(Us[1], us[1]), ()) #ODEs._make_uh_from_us(odeop,us,Us)
