module SingleFieldUtilsFEMTests

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
using Mabla
using Mabla.FEM
using Mabla.Distributed

using PartitionedArrays
using GridapDistributed

import SparseArrays.UMFPACK: UmfpackLU
import Gridap.Helpers: @check
import Gridap.ODEs.TransientFETools: TransientFEOperatorFromWeakForm
import Gridap.ODEs.TransientFETools: get_algebraic_operator
import Gridap.ODEs.TransientFETools: allocate_cache
import Gridap.ODEs.TransientFETools: update_cache!
import Gridap.ODEs.TransientFETools: get_order
import Gridap.ODEs.TransientFETools: _matdata_jacobian

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)

ranks = LinearIndices((4,))
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
order = 1
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,2*order)

t0 = 0
dt = 0.005
θ = 0.5

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = PTFunction(a,μ,t)
f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = PTFunction(f,μ,t)
h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = PTFunction(h,μ,t)
g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = PFunction(u0,μ)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialPFESpace(test,g)
trial0 = trial(nothing,nothing)
pspace = ParametricSpace(fill([1.,10.],3))
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial0)

res(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v) = ∫(v*dut)dΩ

feop = AffineTransientPFEOperator(res,jac,jac_t,pspace,trial,test)

function compute_xh(
  feop::TransientPFEOperator,
  params::Table,
  times::Vector{<:Real},
  u::Tuple)

  ode_cache = allocate_cache(feop,params,times)
  update_cache!(ode_cache,feop,params,times)
  Us, = ode_cache

  uh = EvaluationFunction(Us[1],u[1])
  dxh = ()
  for i in 2:FEM.get_order(feop)+1
    dxh = (dxh...,EvaluationFunction(Us[i],u[i]))
  end

  return TransientCellField(uh,dxh)
end

function get_feoperator_gridap(
  feop::TransientPFEOperator{Affine},
  μ::Vector{<:Real})

  trial_μt = get_trial(feop)
  g(x,t) = trial_μt.dirichlet_μt(x,μ,t)
  g(t) = x->g(x,t)
  trial_t = TransientTrialFESpace(feop.test,g)

  m(t,dut,v) = ∫(v*dut)dΩ
  lhs(t,du,v) = ∫(a(μ,t)*∇(v)⋅∇(du))dΩ
  rhs(t,v) = ∫(f(μ,t)*v)dΩ + ∫(h(μ,t)*v)dΓn

  return TransientAffineFEOperator(m,lhs,rhs,trial_t,test)
end

function compute_xh_gridap(
  feop::TransientFEOperator,
  t::Real,
  u::Tuple)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op)
  update_cache!(ode_cache,ode_op,t)
  Uh, = ode_cache
  uh = EvaluationFunction(Uh[1],u[1])
  dxh = ()
  for i in 2:get_order(feop)+1
    dxh = (dxh...,EvaluationFunction(Uh[i],u[i]))
  end
  TransientCellField(uh,dxh)
end

function compute_res_gridap(
  feop::TransientFEOperator,
  t::Real,
  u::Tuple,
  trian=Ω)

  dv = get_fe_basis(test)
  xh = compute_xh_gridap(feop,t,u)
  res_contrib = feop.res(t,xh,dv)
  vecdata = collect_cell_vector(feop.test,res_contrib,trian)
  assemble_vector(feop.assem_t,vecdata)
end

function compute_jac_gridap(
  feop::TransientFEOperator,
  t::Real,
  u::Tuple,
  trian=Ω,
  i=1)

  _trial0 = feop.trials[1](nothing)
  dv = get_fe_basis(test)
  du = get_trial_fe_basis(_trial0)
  xh = compute_xh_gridap(feop,t,u)
  γᵢ = (1,1/(θ*dt))[i]
  jac_contrib = γᵢ*feop.jacs[i](t,xh,du,dv)
  matdata = collect_cell_matrix(_trial0,test,jac_contrib,trian)
  assemble_matrix(feop.assem_t,matdata)
end

function check_ptarray(a,b;n=1)
  @check all(map(x->x[n],a) .== b) "Detected difference in value for index $n"
end

function Base.isapprox(A::UmfpackLU,B::UmfpackLU)
  A.colptr == B.colptr && A.rowval == B.rowval && A.nzval ≈ B.nzval
end

export model
export Ω,Γn
export dΩ,dΓn
export t0,dt,θ
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
export compute_res_gridap
export compute_jac_gridap
export check_ptarray

end # module
