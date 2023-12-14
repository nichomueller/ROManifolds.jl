# using Gridap
# using GridapDistributed
# using PartitionedArrays

# function main(ranks)
#   domain = (0,1,0,1)
#   mesh_partition = (2,2)
#   mesh_cells = (4,4)
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 2
#   u((x,y)) = (x+y)^order
#   f(x) = -Δ(u,x)
#   reffe = ReferenceFE(lagrangian,Float64,order)
#   V = TestFESpace(model,reffe,dirichlet_tags="boundary")
#   U = TrialFESpace(u,V)
#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,2*order)
#   a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
#   l(v) = ∫( v*f )dΩ
#   op = AffineFEOperator(a,l,U,V)
#   uh = solve(op)
#   writevtk(Ω,"results",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
# end

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   main(ranks)
# end

using Gridap
using Gridap.FESpaces
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
nsnaps_state = 50
nsnaps_mdeim = 20
nsnaps_test = 10
st_mdeim = true
rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

#   order = 1
#   degree = 2*order
#   Ω = Triangulation(model)
#   Γn = BoundaryTriangulation(model,tags=[7,8])
#   dΩ = Measure(Ω,degree)
#   dΓn = Measure(Γn,degree)

#   ranges = fill([1.,10.],3)
#   pspace = PSpace(ranges)

#   a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
#   a(μ,t) = x->a(x,μ,t)
#   aμt(μ,t) = PTFunction(a,μ,t)

#   f(x,μ,t) = 1.
#   f(μ,t) = x->f(x,μ,t)
#   fμt(μ,t) = PTFunction(f,μ,t)

#   h(x,μ,t) = abs(cos(t/μ[3]))
#   h(μ,t) = x->h(x,μ,t)
#   hμt(μ,t) = PTFunction(h,μ,t)

#   g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
#   g(μ,t) = x->g(x,μ,t)

#   u0(x,μ) = 0
#   u0(μ) = x->u0(x,μ)
#   u0μ(μ) = PFunction(u0,μ)

#   res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
#   jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
#   jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialPFESpace(test,g)
#   feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
#   t0,tf,dt,θ = 0.,0.3,0.005,0.5
#   uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
#   fe = trial(Table([rand(3) for _ = 1:3]),[t0,dt])
#   println(typeof(fe))
#   fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
#   uh0μ(Table([rand(3) for _ = 1:3]))
#   error("stop")

#   sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
#   # rbspace = reduced_basis(rbinfo,feop,sols)
#   # rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
# end

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

res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialPFESpace(test,g)
feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

nparams = 2
params = realization(feop,nparams)
w0 = get_free_dof_values(uh0μ(params))
time_ndofs = num_time_dofs(fesolver)
T = get_vector_type(feop.test)
sol = PODESolution(fesolver,feop,params,w0,t0,tf)
Base.iterate(sol)

TODO
# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 1

#   g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
#   g(μ,t) = x->g(x,μ,t)
#   u0(x,μ) = 0
#   u0(μ) = x->u0(x,μ)
#   u0μ(μ) = PFunction(u0,μ)

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialPFESpace(test,g)
#   t0 = 0.
#   uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
#   uh0μ(Table([rand(3) for _ = 1:3]))
# end

# with_debug() do distribute
#   ranks = distribute(LinearIndices((4,)))
#   model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
#   order = 1

#   g(x,t) = exp(-x[1])*abs(sin(t))
#   g(t) = x->g(x,t)
#   u0(x) = 0

#   T = Float64
#   reffe = ReferenceFE(lagrangian,T,order)
#   test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
#   trial = TransientTrialFESpace(test,g)
#   t0 = 0.
#   uh0μ = interpolate_everywhere(u0,trial(t0))
# end

ranks = LinearIndices((4,))
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
order = 2
u((x,y)) = (x+y)^order
f(x) = -Δ(u,x)
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe,dirichlet_tags="boundary")
U = TrialFESpace(u,V)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
l(v) = ∫( v*f )dΩ
op = AffineFEOperator(a,l,U,V)
uh = solve(op)
