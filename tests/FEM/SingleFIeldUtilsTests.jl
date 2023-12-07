module SingleFieldUtilsTests

using LinearAlgebra
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
using Mabla
using Mabla.FEM

import Gridap.Helpers: @check
import Gridap.ODEs.TransientFETools: get_algebraic_operator,allocate_cache,update_cache!

root = pwd()
model = DiscreteModelFromFile(joinpath(root,"models/cube2x2.json"))
order = 1
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = PTFunction(a,μ,t)
f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = PTFunction(f,μ,t)
h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = PTFunction(h,μ,t)
g(μ,x,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(μ,x,t)
u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = PFunction(u0,μ)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialPFESpace(test,g)
trial0 = trial(nothing,nothing)
pspace = PSpace(fill([1.,10.],3))
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial0)

res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)

function compute_xh(feop::PTFEOperator,params::Table,times::Vector{<:Real})
  N = length(times)*length(params)
  u = PTArray([ones(num_free_dofs(feop.test)) for _ = 1:N])

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,params,times)
  update_cache!(ode_cache,ode_op,params,times)
  Us, = ode_cache
  uh = EvaluationFunction(Us[1],u)
  dxh = ()
  for _ in 1:get_order(feop)
    dxh = (dxh...,uh)
  end

  return TransientCellField(uh,dxh)
end

function get_feoperator_gridap(
  feop::PTFEOperator{T},
  μ::Vector{<:Real}) where T

  trial_μt = get_trial(feop)
  g(x,t) = trial_μt.dirichlet_μt(x,μ,t)
  g(t) = x->g_ok(x,t)
  trial_t = TransientTrialFESpace(feop.test,g)

  rhs(t,u,dv) = feop.rhs(μ,t,u,dv)
  lhs(t,u,du,dv) = feop.lhs[1](μ,t,u,du,dv)
  lhs_t(t,u,dut,dv) = feop.lhs[2](μ,t,u,dut,dv)

  return TransientFEOperatorFromWeakForm{T}(
    rhs,
    (lhs,lhs_t),
    feop.test,
    feop.assem,
    (trial_t,∂t(trial_t)),
    1)
end

function compute_xh_gridap(feop::TransientFEOperator,t::Real)
  odeop = get_algebraic_operator(feop)
  u = ones(num_free_dofs(feop.test))
  ode_cache = allocate_cache(odeop)
  update_cache!(ode_cache,odeop,t)
  Uh, = ode_cache
  uh_ok = EvaluationFunction(Uh[1],u)
  dxh_ok = ()
  for _ in 1:get_order(feop)
    dxh_ok = (dxh_ok...,uh_ok)
  end
  TransientCellField(uh_ok,dxh_ok)
end

function check_ptarray(a,b;n=1)
  @check all(map(x->x[n],a) .== b) "Detected difference in value for index $n"
end

export model
export Ω
export dΩ
export a,aμt,f,fμt,h,hμt,g,res,jac,jac_t,u0μ,uh0μ
export test
export trial
export trial0
export dv
export du
export feop
export compute_xh
export get_feoperator_gridap
export compute_xh_gridap
export check_ptarray

end # module
