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

import Base: inv
import Base: abs
import Base: abs2
import Base: *
import Base: +
import Base: -
import Base: /
import Base: adjoint
import Base: transpose
import Base: real
import Base: imag
import Base: conj
import LinearAlgebra: det
import LinearAlgebra: tr
import LinearAlgebra: cross
import LinearAlgebra: dot
import LinearAlgebra: fillstored!
import BlockArrays: blockedrange
import FillArrays: Fill
import FillArrays: fill
import Distributions: Uniform
import Distributions: Normal
import ForwardDiff: derivative
import UnPack: @unpack
import Gridap.Helpers: @abstractmethod
import Gridap.Helpers: @check
import Gridap.Helpers: @notimplemented
import Gridap.Helpers: @unreachable
import Gridap.Algebra: InserterCSC
import Gridap.CellData: CellField
import Gridap.CellData: GenericMeasure
import Gridap.CellData: CompositeMeasure
import Gridap.CellData: DomainStyle
import Gridap.CellData: OperationCellField
import Gridap.CellData: change_domain
import Gridap.CellData: similar_cell_field
import Gridap.CellData: _get_cell_points
import Gridap.CellData: _operate_cellfields
import Gridap.CellData: _to_common_domain
import Gridap.Fields: OperationField
import Gridap.Fields: BroadcastOpFieldArray
import Gridap.Fields: BroadcastingFieldOpMap
import Gridap.Fields: LinearCombinationField
import Gridap.Fields: LinearCombinationMap
import Gridap.FESpaces: FEFunction
import Gridap.FESpaces: SparseMatrixAssembler
import Gridap.FESpaces: EvaluationFunction
import Gridap.FESpaces: _pair_contribution_when_possible
import Gridap.MultiField: MultiFieldFEBasisComponent
import Gridap.ReferenceFEs: get_order
import Gridap.ODEs.ODETools: residual!
import Gridap.ODEs.ODETools: jacobian!
import Gridap.ODEs.ODETools: jacobians!
import Gridap.ODEs.ODETools: _allocate_matrix_and_vector
import Gridap.ODEs.TransientFETools: ODESolver
import Gridap.ODEs.TransientFETools: ODEOperator
import Gridap.ODEs.TransientFETools: OperatorType
import Gridap.ODEs.TransientFETools: TransientCellField
import Gridap.ODEs.TransientFETools: TransientSingleFieldCellField
import Gridap.ODEs.TransientFETools: TransientMultiFieldCellField
import Gridap.ODEs.TransientFETools: TransientFEBasis
import Gridap.ODEs.TransientFETools: SingleFieldTypes
import Gridap.ODEs.TransientFETools: MultiFieldTypes
import Gridap.ODEs.TransientFETools: allocate_trial_space
import Gridap.ODEs.TransientFETools: fill_jacobians
import Gridap.ODEs.TransientFETools: _matdata_jacobian
import Gridap.ODEs.TransientFETools: _vcat_matdata
import Gridap.TensorValues: inner
import Gridap.TensorValues: outer
import Gridap.TensorValues: double_contraction
import Gridap.TensorValues: symmetric_part
import PartitionedArrays: tuple_of_arrays

include("../../src/FEM/ParametricSpace.jl")
include("../../src/FEM/PDiffOperators.jl")
include("../../src/FEM/PArray.jl")
include("../../src/FEM/PField.jl")
include("../../src/FEM/TrialPFESpace.jl")
include("../../src/FEM/TransientTrialPFESpace.jl")
include("../../src/FEM/PCellField.jl")
include("../../src/FEM/PAssemblers.jl")
include("../../src/FEM/TransientPFEOperator.jl")
include("../../src/FEM/PODEOperatorInterface.jl")
include("../../src/FEM/PTSolvers.jl")
include("../../src/FEM/PThetaMethod.jl")
include("../../src/FEM/PVisualization.jl")
include("../../src/FEM/FEUtils.jl")
include("../../src/FEM/ReducedMeasure.jl")

root = pwd()
model = DiscreteModelFromFile(joinpath(root,"models/elasticity_3cyl2D.json"))
test_path = "$root/results/HeatEquation/elasticity_3cyl2D"
order = 1
degree = 2*order
Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

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

res(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v) = ∫(v*dut)dΩ

pranges = fill([1.,10.],3)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
tdomain = t0:dt:tf
ptspace = TransientParametricSpace(pranges,tdomain)

T = Float
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialPFESpace(test,g)
feop = AffineTransientPFEOperator(res,jac,jac_t,ptspace,trial,test)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),θ,dt)

ϵ = 1e-4
load_solutions = false
save_solutions = true
load_structures = false
save_structures = true
postprocess = true
norm_style = :l2
nsnaps_state = 50
nsnaps_mdeim = 20
nsnaps_test = 10
st_mdeim = false
rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
rbspace = reduced_basis(rbinfo,feop,sols)



abstract type ReducedFESpace <: FESpace end
struct ReducedSingleFieldFESpace{F,R} <: ReducedFESpace
  fe::F
  reduced_basis::R
end



w = (u*v)
cache = return_cache(w,x)
@which evaluate!(cache,w,x)
u(x)

boh = ∫(a(rand(3),dt)*∇(φ)⋅∇(φ))dΩ
boh[Ω]

φᵢ = FEFunction(test,bs1)
φⱼ = FEFunction(test,bs1)
@time for bsi in eachcol(bs)
  for bsj in eachcol(bs)
    ∫(a(rand(3),dt)*∇(φᵢ)⋅∇(φⱼ))dΩ
  end
end

trial0 = trial(nothing,nothing)
@time begin
  μ = rand(3)
  A = assemble_matrix((φᵢ,φⱼ)->∫(a(μ,dt)*∇(φᵢ)⋅∇(φⱼ))dΩ,trial0,test)
  bs'*A*bs
end

(φᵢ*φᵢ)(x)
fs,free_values,dirichlet_values = test,bs1,get_dirichlet_dof_values(test)
cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
cell_field = CellField(fs,cell_vals)
SingleFieldFEFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)

struct DummyFunction
end


𝒯 = CartesianDiscreteModel((0,1,0,1),(20,20))
Ω = Interior(𝒯)
dΩ = Measure(Ω,2)
refFE = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(𝒯,refFE,dirichlet_tags="boundary")
g(x,t::Real) = 0.0
g(t::Real) = x -> g(x,t)
U = TransientTrialFESpace(V,g)
κ(t) = 1.0 + 0.95*sin(2π*t)
f(t) = sin(π*t)
res(t,u,v) = ∫( ∂t(u)*v + κ(t)*(∇(u)⋅∇(v)) - f(t)*v )dΩ
jac(t,u,du,v) = ∫( κ(t)*(∇(du)⋅∇(v)) )dΩ
jac_t(t,u,duₜ,v) = ∫( duₜ*v )dΩ
op = TransientFEOperator(res,jac,jac_t,U,V)
m(t,u,v) = ∫( u*v )dΩ
a(t,u,v) = ∫( κ(t)*(∇(u)⋅∇(v)) )dΩ
b(t,v) = ∫( f(t)*v )dΩ
op_Af = TransientAffineFEOperator(m,a,b,U,V)
linear_solver = LUSolver()
Δt = 0.05
θ = 0.5
ode_solver = ThetaMethod(linear_solver,Δt,θ)
u₀ = interpolate_everywhere(0.0,U(0.0))
t₀ = 0.0
T = 10.0
uₕₜ = solve(ode_solver,op,u₀,t₀,T)
using Gridap.Visualization
import Gridap.Visualization: _prepare_cdata,_prepare_pdata
# function Visualization._prepare_pdata(trian,cellfields,samplingpoints)
#   println(typeof(cellfields))
#   x = CellPoint(samplingpoints,trian,ReferenceDomain())
#   pdata = Dict()
#   for (k,v) in cellfields
#     _v = CellField(v,trian)
#     pdata[k], = Visualization._prepare_node_to_coords(evaluate(_v,x))
#   end
#   pdata
# end
# createpvd("poisson_transient_solution") do pvd
#   for (uₕ,t) in uₕₜ
#     pvd[t] = createvtk(Ω,"poisson_transient_solution_$t"*".vtu",cellfields=["u"=>uₕ])
#   end
# end


ff = FEFunction(V,rand(num_free_dofs(V)))
writevtk(Ω,"test",cellfields=["u"=>ff])

x = rand(num_free_dofs(test))
pa = parray(x,2)
pff = FEFunction(trial([rand(3),rand(3)],dt),pa)
ppath = PString("test",2)
# vd = Gridap.Visualization.visualization_data(Ω,ppath,cellfields=Dict("u"=>pff))
writevtk(Ω,ppath,cellfields=Dict("u"=>pff))

trian = Ω
ref_grids = map((reffe) -> UnstructuredGrid(reffe),get_reffes(trian))
visgrid = Visualization.VisualizationGrid(trian,ref_grids)

cdata = _prepare_cdata(Dict(),visgrid.sub_cell_to_cell)
cellfields=["u"=>pff]
# pdata = _prepare_pdata(trian,cellfields,visgrid.cell_to_refpoints)
x = CellPoint(visgrid.cell_to_refpoints,trian,ReferenceDomain())
pdata = Dict()
_v = CellField(pff,trian)

# pdata["u"], = Visualization._prepare_node_to_coords(evaluate(_v,x))
cell_to_points=evaluate(_v,x)
cell_to_offset = zeros(Int,length(cell_to_points))
P = eltype(eltype(cell_to_points))
node_to_coords = P[]
cache = array_cache(cell_to_points)
