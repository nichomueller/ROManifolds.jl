include("./SingleFieldUtilsFEMTests.jl")

module SolutionTests

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

using GridapDistributed

using Mabla
using Mabla.FEM

using Main.SingleFieldUtilsFEMTests

import Gridap.Helpers: @check
import Gridap.ODEs.TransientFETools: get_algebraic_operator,GenericODESolution

ntimes = 3
tf = (ntimes-1)*dt
nparams = 2
params = realization(feop,nparams)

fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
w = get_free_dof_values(uh0μ(params))
sol = PODESolution(fesolver,feop,params,w,t0,tf)

results = []
for (uh,t) in sol
  push!(results,copy(uh))
end

for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  ode_op_t = get_algebraic_operator(feop_t)
  ode_solver = ThetaMethod(LUSolver(),dt,θ)
  w_t = zero_free_values(test)
  sol_t = GenericODESolution(ode_solver,ode_op_t,w_t,t0,tf)

  results_t = []
  for (uh,t) in sol_t
    push!(results_t,copy(uh))
  end

  for (α,β) in zip(results,results_t)
    map(local_views(α),local_views(β)) do α,β
      @check isapprox(α[np],β) "Detected difference in value for index $np"
    end
  end
end
end # module

module DebugArraySolutionTest
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

using GridapDistributed
using PartitionedArrays

using Mabla
using Mabla.FEM

using Main.SingleFieldUtilsFEMTests

import Gridap.Helpers: @check
import Gridap.ODEs.TransientFETools: get_algebraic_operator,GenericODESolution

ntimes = 3
tf = (ntimes-1)*dt
nparams = 2
params = realization(feop,nparams)

function main(ranks)
  domain = (0,1,0,1)
  mesh_partition = (2,2)
  mesh_cells = (4,4)

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
  pspace = PSpace(fill([1.,10.],3))

  res(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
  jac(μ,t,u,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
  jac_t(μ,t,u,dut,v) = ∫(v*dut)dΩ

  feop = AffineTransientPFEOperator(res,jac,jac_t,pspace,trial,test)
  fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
  w = get_free_dof_values(uh0μ(params))
  sol = PODESolution(fesolver,feop,params,w,t0,tf)

  results = []
  for (uh,t) in sol
    push!(results,copy(uh))
  end

  return results
end

function main_gridap(ranks,μ)
  domain = (0,1,0,1)
  mesh_partition = (2,2)
  mesh_cells = (4,4)

  model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
  order = 1
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  Γn = BoundaryTriangulation(model,tags=[7,8])
  dΓn = Measure(Γn,2*order)

  t0 = 0
  dt = 0.005
  θ = 0.5

  a(x,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(t) = x->a(x,t)
  f(x,t) = 1.
  f(t) = x->f(x,t)
  h(x,t) = abs(cos(t/μ[3]))
  h(t) = x->h(x,t)
  g(x,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  g(t) = x->g(x,t)
  u0(x) = 0

  T = Float64
  reffe = ReferenceFE(lagrangian,T,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
  trial = TransientTrialFESpace(test,g)

  m(t,dut,v) = ∫(v*dut)dΩ
  lhs(t,du,v) = ∫(a(t)*∇(v)⋅∇(du))dΩ
  rhs(t,v) = ∫(f(t)*v)dΩ + ∫(h(t)*v)dΓn

  feop = TransientAffineFEOperator(m,lhs,rhs,trial,test)
  ode_solver = ThetaMethod(LUSolver(),dt,θ)
  uh0 = interpolate_everywhere(u0,trial(t0))
  w = get_free_dof_values(uh0)
  ode_op = get_algebraic_operator(feop)
  sol = GenericODESolution(ode_solver,ode_op,w,t0,tf)

  results = []
  for (uh,t) in sol
    push!(results,copy(uh))
  end

  return results
end

with_debug() do distribute
  ranks = distribute(LinearIndices((4,)))
  results = main(ranks)
  for np = eachindex(params)
    results_t = main_gridap(ranks,params[np])
    for (α,β) in zip(results,results_t)
      map(local_views(α),local_views(β)) do α,β
        @check isapprox(α[np],β) "Detected difference in value for index $np"
      end
    end
  end
end
end # module
