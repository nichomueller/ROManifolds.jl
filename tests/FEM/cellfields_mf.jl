using Distributions: Normal, Uniform
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
order = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = PTFunction(a,μ,t)
g(μ,x,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(μ,x,t)
g0(x,μ,t) = VectorValue(0,0)
g0(μ,t) = x->g0(x,μ,t)
u0(x,μ) = VectorValue(0,0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = PFunction(u0,μ)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

T = Float
reffe_u = ReferenceFE(lagrangian,VectorValue{2,T},order)
reffe_p = ReferenceFE(lagrangian,T,order-1)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial_u = TransientTrialPFESpace(test_u,[g0,g])
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldPFESpace([test_u,test_p])
trial = TransientMultiFieldPFESpace([trial_u,trial_p])
trial0 = trial(nothing,nothing)
trial0_u = trial_u(nothing,nothing)
pspace = PSpace(fill([1.,10.],3))
dv = get_fe_basis(test_u)
du = get_trial_fe_basis(trial0_u)

res(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)

feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)

ntimes = 3
nparams = 2
times = rand(ntimes)
params = realization(feop,nparams)

x = get_cell_points(Ω)

cf_mat = aμt(params,times)*∇(dv)⊙∇(du)
cfx_mat = cf_mat(x)

for np in 1:nparams
  for nt in 1:ntimes
    cf_mat_t = a(params[np],times[nt])*∇(dv)⋅∇(du)
    cfx_mat_t = cf_mat_t(x)
    check_ptarray(cfx_mat,cfx_mat_t;n = (np-1)*ntimes+nt)
  end
end
