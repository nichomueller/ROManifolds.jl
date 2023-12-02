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
include("/home/nicholasmueller/git_repos/Mabla.jl/src/Utils/Indexes.jl")
include("/home/nicholasmueller/git_repos/Mabla.jl/src/FEM/NewFile.jl")

g(x,t) = x[1]*t
g(t) = x->g(x,t)

root = pwd()
model = DiscreteModelFromFile(joinpath(root,"models/cube2x2.json"))
order = 1
T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialFESpace(test,g)
t0,tf,dt,θ = 0.,0.3,0.005,0.5

μ = [rand(3) for _ = 1:2]
times = [dt,2dt,3dt]
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing))

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = PTFunction(a,μ,t)

trian = Triangulation(model)
cf = aμt(μ,times)*∇(dv)⋅∇(du)
x = get_cell_points(trian)
result = cf(x)
cf_ok = a(μ[1],times[1])*∇(dv)⋅∇(du)
result_ok = cf_ok(x)

for k in 1:2
  for m in 1:3
    b1 = CellField(a(μ[k],times[m]),trian,PhysicalDomain())
    q1 = evaluate!(nothing,b1,x)
    n = (k-1)*3+m
    mycheck(q2,q1;n=1)
  end
end

for k in 1:2
  for m in 1:3
    cf_ok = a(μ[k],times[m])*∇(dv)⋅∇(du)
    cfx_ok = cf_ok(x)
    ccfx_ok = collect(cfx_ok)
    n = (k-1)*3+m
    @assert all(map(x->x[n],ccfx) .== ccfx_ok) "Detected difference in value for index $n"
  end
end
