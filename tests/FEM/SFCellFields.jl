using LinearAlgebra
import FillArrays: Fill
import FillArrays: fill
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
import Base: *
import Base: +
import Base: -
import Base: /
import ForwardDiff: derivative
import LinearAlgebra: fillstored!
import Gridap.Helpers: @check,@unreachable,@abstractmethod,@notimplemented
import Gridap.Fields: BroadcastingFieldOpMap
import Gridap.Fields: LinearCombinationField, LinearCombinationMap
import Gridap.CellData: OperationCellField
import Gridap.CellData: _get_cell_points
import Gridap.CellData: _operate_cellfields
import Gridap.CellData: _to_common_domain
import Gridap.CellData: GenericMeasure
import Gridap.ODEs.TransientFETools: allocate_trial_space
import Gridap.ODEs.TransientFETools: jacobians!, fill_jacobians, _matdata_jacobian, _vcat_matdata
import Gridap.ODEs.TransientFETools: TransientSingleFieldCellField
const Float = Float64
include("../../src/Utils/Indexes.jl")
include("../../src/FEM/PSpace.jl")
include("../../src/FEM/PDiffOperators.jl")
include("../../src/FEM/PTArray.jl")
include("../../src/FEM/PTFields.jl")
include("../../src/FEM/PFESpaces.jl")
include("../../src/FEM/PTFESpaces.jl")
include("../../src/FEM/PTCellFields.jl")
include("../../src/FEM/PTIntegrand.jl")
include("../../src/FEM/PTAssemblers.jl")
include("../../src/FEM/PTFEOperator.jl")
include("../../src/FEM/PODEOperatorInterface.jl")
include("../../src/FEM/PTOperator.jl")
include("../../src/FEM/PTSolvers.jl")
include("../../src/FEM/PAffineThetaMethod.jl")

root = pwd()
model = DiscreteModelFromFile(joinpath(root,"models/cube2x2.json"))
order = 1

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = PTFunction(a,μ,t)
g(x,t) = x[1]*t
g(t) = x->g(x,t)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialFESpace(test,g)
trial0 = trial(nothing)

μ = [rand(3) for _ = 1:2]
times = [1,2,3]
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing))

trian = Triangulation(model)
x = get_cell_points(trian)
cf = aμt(μ,times)*∇(dv)⋅∇(du)
cfx = cf(x)

function _check(a,b;n=1)
  @assert all(map(x->x[n],a) .== b) "Detected difference in value for index $n"
end

for k in 1:2
  for m in 1:3
    cf_ok = a(μ[k],times[m])*∇(dv)⋅∇(du)
    cfx_ok = cf_ok(x)
    _check(cfx,cfx_ok;n = (k-1)*3+m)
  end
end

dΩ = Measure(trian,2)
int = ∫(aμt(μ,times)*∇(dv)⋅∇(du))dΩ

for k in 1:2
  for m in 1:3
    int_ok = ∫(a(μ[k],times[m])*∇(dv)⋅∇(du))dΩ
    _check(int[trian],int_ok[trian];n = (k-1)*3+m)
  end
end

matdata = collect_cell_matrix(trial0,test,int)
global matdata_ok
for k in 1:2
  for m in 1:3
    int_ok = ∫(a(μ[k],times[m])*∇(dv)⋅∇(du))dΩ
    matdata_ok = collect_cell_matrix(trial0,test,int_ok)
    _check(matdata[1][1],matdata_ok[1][1];n = (k-1)*3+m)
  end
end
@check matdata[2] == matdata_ok[2]
@check matdata[3] == matdata_ok[3]

assem = SparseMatrixAssembler(trial0,test)
A_ok = allocate_matrix(assem,matdata_ok)
A = PTArray([copy(A_ok) for _ = 1:6])
assemble_matrix_add!(A,assem,matdata)
for k in 1:2
  for m in 1:3
    int_ok = ∫(a(μ[k],times[m])*∇(dv)⋅∇(du))dΩ
    matdata_ok = collect_cell_matrix(trial0,test,int_ok)
    A_ok = allocate_matrix(assem,matdata_ok)
    assemble_matrix_add!(A_ok,assem,matdata_ok)
    @check A[(k-1)*3+m] == A_ok "Detected difference in value for index $((k-1)*3+m)"
  end
end

Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,2)

gμ(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμ(μ,t) = x->gμ(x,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = PTFunction(f,μ,t)
h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = PTFunction(h,μ,t)
res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)
pspace = PSpace(fill([1.,10.],3))
trial = PTTrialFESpace(test,gμ)
feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

N = length(times)*length(μ)
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
update_cache!(ode_cache,ode_op,μ,times)
nfree = num_free_dofs(test)
u = PTArray([zeros(nfree) for _ = 1:N])
vθ = similar(u)
vθ .= 1.0
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(feop)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

function compute_xh_gridap(k,m)
  g_ok(x,t) = gμ(x,μ[k],t)
  g_ok(t) = x->g_ok(x,t)
  trial_ok = TransientTrialFESpace(test,g_ok)
  du_ok = get_trial_fe_basis(trial_ok(times[m]))
  m_ok(t,dut,dv) = ∫(dv*dut)dΩ
  lhs_ok(t,du,dv) = ∫(a(μ[k],t)*∇(dv)⋅∇(du))dΩ
  rhs_ok(t,dv) = ∫(f(μ[k],t)*dv)dΩ + ∫(h(μ[k],t)*dv)dΓn
  feop_ok = TransientAffineFEOperator(m_ok,lhs_ok,rhs_ok,trial_ok,test)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  vθ_ok = similar(u[1])
  vθ_ok .= 1.0
  ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,times[m])
  Us_ok,_,_ = ode_cache_ok
  uh_ok = EvaluationFunction(Us_ok[1],vθ_ok)
  dxh_ok = ()
  for i in 1:get_order(feop)
    dxh_ok = (dxh_ok...,uh_ok)
  end
  TransientCellField(uh_ok,dxh_ok)
end

cf = aμt(μ,times)*∇(dv)⋅∇(xh)
cfx = cf(x)

for k in 1:2
  for m in 1:3
    xh_ok = compute_xh_gridap(k,m)
    cf_ok = a(μ[k],times[m])*∇(dv)⋅∇(xh_ok)
    cfx_ok = cf_ok(x)
    _check(cfx,cfx_ok;n = (k-1)*3+m)
  end
end

int = ∫(aμt(μ,times)*∇(dv)⋅∇(xh))dΩ

for k in 1:2
  for m in 1:3
    xh_ok = compute_xh_gridap(k,m)
    int_ok = ∫(a(μ[k],times[m])*∇(dv)⋅∇(xh_ok))dΩ
    _check(int[trian],int_ok[trian];n = (k-1)*3+m)
  end
end

vecdata = collect_cell_vector(test,int)
global vecdata_ok
for k in 1:2
  for m in 1:3
    xh_ok = compute_xh_gridap(k,m)
    int_ok = ∫(a(μ[k],times[m])*∇(dv)⋅∇(xh_ok))dΩ
    vecdata_ok = collect_cell_vector(test,int_ok)
    _check(vecdata[1][1],vecdata_ok[1][1];n = (k-1)*3+m)
  end
end
@check vecdata[2] == vecdata_ok[2]

b_ok = allocate_vector(assem,vecdata_ok)
b = PTArray([copy(b_ok) for _ = 1:6])
assemble_vector_add!(b,assem,vecdata)
for k in 1:2
  for m in 1:3
    xh_ok = compute_xh_gridap(k,m)
    int_ok = ∫(a(μ[k],times[m])*∇(dv)⋅∇(xh_ok))dΩ
    vecdata_ok = collect_cell_vector(test,int_ok)
    b_ok = allocate_vector(assem,vecdata_ok)
    assemble_vector_add!(b_ok,assem,vecdata_ok)
    @check b[(k-1)*3+m] == b_ok "Detected difference in value for index $((k-1)*3+m)"
  end
end

r(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
r_ok(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ - ∫(h(μ,t)*v)dΓn
int = r(μ,times,xh,dv)
for k in 1:2
  for m in 1:3
    xh_ok = compute_xh_gridap(k,m)
    int_ok = r_ok(μ[k],times[m],xh_ok,dv)
    _check(int[trian],int_ok[trian];n = (k-1)*3+m)
    _check(int[Γn],int_ok[Γn];n = (k-1)*3+m)
  end
end

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = PFunction(u0,μ)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
w = get_free_dof_values(uh0μ(μ))
sol = PODESolution(fesolver,ode_op,μ,w,t0,tf)

results = PTArray[]
for (uh,t) in sol
  push!(results,copy(uh))
end

n = 2
p,v = μ[n],w[n]
g_ok(x,t) = gμ(x,p,t)
g_ok(t) = x->g_ok(x,t)
a_ok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
f_ok(t,v) = ∫(v*f(p,t))dΩ + ∫(v*h(p,t))dΓn
m_ok(t,ut,v) = ∫(ut*v)dΩ

trial_ok = TransientTrialFESpace(test,g_ok)
feop_ok = TransientAffineFEOperator(m_ok,a_ok,f_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_solver = ThetaMethod(LUSolver(),dt,θ)
sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,v,t0,tf)

results_ok = Vector{Float}[]
for (uh,t) in sol_gridap
  push!(results_ok,copy(uh))
end

for (α,β) in zip(results,results_ok)
  @check isapprox(α[n],β)
end

odeop = get_algebraic_operator(feop)
ode_cache = allocate_cache(odeop,μ,t0)
vθ = similar(w)
vθ .= 0.0
l_cache = nothing
A,b = _allocate_matrix_and_vector(odeop,μ,t0,vθ,ode_cache)
ode_cache = update_cache!(ode_cache,odeop,μ,t0)
_matrix_and_vector!(A,b,odeop,μ,t0,dt*θ,vθ,ode_cache,vθ)
