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
import Gridap.Helpers: @check,@unreachable,@abstractmethod
import Gridap.Fields: BroadcastingFieldOpMap
import Gridap.CellData: OperationCellField
import Gridap.CellData: _get_cell_points
import Gridap.CellData: _operate_cellfields
import Gridap.CellData: _to_common_domain
import Gridap.CellData: GenericMeasure
const Float = Float64
include("../../src/Utils/Indexes.jl")
include("../../src/FEM/PSpace.jl")
include("../../src/FEM/PDiffOperators.jl")
include("../../src/FEM/PTArray.jl")
include("../../src/FEM/PTFields.jl")
include("../../src/FEM/PTCellFields.jl")
include("../../src/FEM/PTIntegrand.jl")
include("../../src/FEM/PTAssemblers.jl")

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

fs = trial(μ,t0)
object = u0μ(μ)
free_values = Gridap.FESpaces.zero_free_values(fs)
dirichlet_values = Gridap.FESpaces.zero_dirichlet_values(fs)
s = Gridap.FESpaces.get_fe_dof_basis(fs)
trian = Gridap.FESpaces.get_triangulation(s)
cf = CellField(object,trian,DomainStyle(s))
b = Gridap.FESpaces.change_domain(cf,s.domain_style)
_f = Gridap.CellData.get_data(s),Gridap.CellData.get_data(b)
fi = map(Gridap.Arrays.testitem,_f)
b,field = Gridap.Arrays.testargs(evaluate,fi...)[1],Gridap.Arrays.testargs(evaluate,fi...)[2]
# c,x = field,b.nodes
# _cf = map(fi -> Gridap.Arrays.return_cache(fi,x),c.fields)
# lx = map((ci,fi) -> Gridap.Arrays.evaluate!(ci,fi,x),_cf,c.fields)
# ck = Gridap.Arrays.return_cache(c.op,map(Gridap.Arrays.testitem,lx)...)
#   f̃ = c.op
#   x̃ = map(Gridap.Arrays.testitem,lx)
#   fi = testitem(f̃)
#   li = return_cache(fi,x̃...)
#   fix = evaluate!(li,fi,x̃...)
#   l = Vector{typeof(li)}(undef,size(f̃.fields))
#   g = Vector{typeof(fix)}(undef,size(f̃.fields))
#   for i in eachindex(f̃.fields)
#     l[i] = return_cache(f̃.fields[i],x̃)
#   end
#   PTArray(g),l
# r = c.op.(lx...)
# ca = Gridap.Arrays.CachedArray(r)

# sx = size(x)
# Gridap.Arrays.setsize!(ca,sx)
# lx = map((ci,fi) -> evaluate!(ci,fi,x),_cf,c.fields)
# r = ca.array
# for i in eachindex(x)
#   @inbounds r[i] = evaluate!(ck,c.op,map(lxi -> lxi[i], lx)...)
# end
c,cf = Gridap.Arrays.return_cache(b,field)
vals = evaluate!(cf,field,b.nodes)
ndofs = length(b.dof_to_node)
S = eltype(vals)
ncomps = num_components(S)

gμ(x,t) = exp(-x[1])*abs(sin(t))
gμ(t) = x->gμ(x,t)
trialμ = Gridap.TransientTrialFESpace(test,gμ)
fs = trialμ(t0)
objectμ = u0μ(μ[1])
free_valuesμ = Gridap.FESpaces.zero_free_values(fs)
dirichlet_valuesμ = Gridap.FESpaces.zero_dirichlet_values(fs)
sμ = Gridap.FESpaces.get_fe_dof_basis(fs)
cfμ = CellField(objectμ,trian,DomainStyle(sμ))
bμ = Gridap.FESpaces.change_domain(cfμ,s.domain_style)
fμ = Gridap.CellData.get_data(sμ),Gridap.CellData.get_data(bμ)
fμi = map(Gridap.Arrays.testitem,fμ)
bμ,fieldμ = Gridap.Arrays.testargs(evaluate,fμi...)[1],Gridap.Arrays.testargs(evaluate,fμi...)[2]
# cμ,xμ = fieldμ,bμ.nodes
# _cfμ = map(fi -> Gridap.Arrays.return_cache(fi,x),cμ.fields)
# lxμ = map((ci,fi) -> Gridap.Arrays.evaluate!(ci,fi,x),_cfμ,cμ.fields)
# ckμ = Gridap.Arrays.return_cache(cμ.op,map(Gridap.Arrays.testitem,lxμ)...)
# rμ = cμ.op.(lxμ...)
# caμ = Gridap.Arrays.CachedArray(rμ)
cμ,cfμ = Gridap.Arrays.return_cache(bμ,fieldμ)
cacheμ = cμ,cfμ
# Gridap.Arrays.evaluate!(cacheμ,bμ,fieldμ)
valsμ = evaluate!(cfμ,fieldμ,bμ.nodes)
ndofs = length(bμ.dof_to_node)
Sμ = eltype(valsμ)
ncomps = num_components(Sμ)
