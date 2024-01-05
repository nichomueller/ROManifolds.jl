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
using Mabla.FEM
using Mabla.RB
using Mabla.Distributed

root = pwd()
test_path = "$root/results/HeatEquation/cube_2x2.json"
ϵ = 1e-4
load_solutions = true
save_solutions = true
load_structures = false
save_structures = true
postprocess = true
norm_style = :H1
nsnaps_state = 10
nsnaps_mdeim = 2
nsnaps_test = 2
st_mdeim = true
rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)
ranks = LinearIndices((4,))
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

order = 1
degree = 2*order
Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

ranges = fill([1.,10.],3)
pspace = PSpace(ranges)

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

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialPFESpace(test,g)
feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.1,0.005,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

# sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
# rbspace = reduced_basis(rbinfo,feop,sols)

uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
nparams = rbinfo.nsnaps_state+rbinfo.nsnaps_test
params = realization(feop,nparams)
time_ndofs = num_time_dofs(fesolver)
uμt = PODESolution(fesolver,feop,params,get_free_dof_values(uh0μ(params)),t0,tf)
Base.length(x::PODESolution) = Int((x.tf-x.t0)/x.solver.dt)
stats = @timed begin
  snaps = map(uμt) do (snap,n)
    copy(snap)
  end
end
Snapshots(snaps)
# sols = Snapshots(snaps)
# rbspace = reduced_basis(rbinfo,feop,sols)

x = snaps[1]
values = map(local_views(x)) do x
  x[1]
end
y = PVector(values,x.index_partition)

collect(y)
collect(x)

_snaps = map(snaps) do a
  b = local_views(a)
end

A = [rand(3) for _ = 1:4]
loc_len = length(first(A))
B = map(i->getindex.(A,i),1:loc_len)

idx = map(part_id,snaps[1].index_partition)
_snaps = map(idx) do i
  map(snaps) do a
    b = local_views(a)
    b[i]
  end
end
