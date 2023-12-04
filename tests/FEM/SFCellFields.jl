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
import Gridap.Helpers: @check,@unreachable
import Gridap.Fields: BroadcastingFieldOpMap
import Gridap.CellData: OperationCellField
import Gridap.CellData: _get_cell_points
import Gridap.CellData: _operate_cellfields
import Gridap.CellData: _to_common_domain
import Gridap.CellData: GenericMeasure
const Float = Float64
include("../../src/Utils/Indexes.jl")
# include("../../src/FEMnew/NewFile.jl")
include("../../src/FEMnew/FEMnew.jl")

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
