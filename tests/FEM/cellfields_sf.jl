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
import Gridap.Helpers: @check
import Gridap.Fields: BroadcastingFieldOpMap
include("/home/nicholasmueller/git_repos/Mabla.jl/src/FEM/NewFile.jl")

root = pwd()
model = DiscreteModelFromFile(joinpath(root,"models/cube2x2.json"))
order = 1
T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
t0,tf,dt,θ = 0.,0.3,0.005,0.5

μ = [rand(3) for _ = 1:2]
times = [dt,2dt,3dt]
dv = get_fe_basis(test)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = PTFunction(a,μ,t)

cf = aμt(μ,times)*∇(dv)
x = get_cell_points(trian)
result = cf(x)
cf_ok = a(μ[1],times[1])*∇(dv)#⋅∇(du)
result_ok = cf_ok(x)
test_ptarray(result,result_ok)

trian = Triangulation(model)
cell_map = get_cell_map(trian)

b1 = CellField(a(μ[1],times[1]),trian,PhysicalDomain())
cell_field_phys1 = get_data(b1)
# cell_field_ref1 = lazy_map(Broadcasting(∘),cell_field_phys1,cell_map)
f1 = map(testitem,(cell_field_phys1,cell_map))
T1 = return_type(Broadcasting(∘),f1...)
lazy_map(Broadcasting(∘),T1,cell_field_phys1,cell_map)

b2 = CellField(aμt(μ,times),trian,PhysicalDomain())
cell_field_phys2 = get_data(b2)
# cell_field_ref2 = lazy_map(Broadcasting(∘),cell_field_phys2,cell_map)
testitem(cell_field_phys2)
f2 = map(testitem,(cell_field_phys2,cell_map))
T2 = return_type(Broadcasting(∘),f2...)
lazy_map(Broadcasting(∘),T2,cell_field_phys2,cell_map)
