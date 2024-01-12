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
load_solutions = false
save_solutions = false
load_structures = false
save_structures = false
postprocess = false
norm_style = :l2
nsnaps_state = 10
nsnaps_mdeim = 2
nsnaps_test = 2
st_mdeim = true
rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(4),)))
end
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
feop = AffineTransientPFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.1,0.01,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

# sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
params = realization(feop,nsnaps_state+nsnaps_test)
w0 = get_free_dof_values(fesolver.uh0(params))
time_ndofs = num_time_dofs(fesolver)
uμt = PODESolution(fesolver,feop,params,w0,fesolver.t0,fesolver.tf)
println("Computing fe solution: time marching across $time_ndofs instants, for $nparams parameters")
stats = @timed begin
  snaps = map(uμt) do (snap,n)
    copy(snap)
  end
end
# sols = Snapshots(snaps)
_type(a::PVector{V}) where V = V
s1 = first(snaps)
S = _type(s1)
parts = map(part_id,s1.index_partition)
snap_parts = map(parts) do part
  cache = S[]
  for si in snaps
    map(local_views(si),si.index_partition) do sij,j
      if j == part
        push!(cache,sij)
      end
    end
    cache
  end
  Snapshots(cache)
end
DistributedSnapshots(snap_parts)
rbspace = reduced_basis(rbinfo,feop,sols)
x = sols[1]
xrec = project_recast(x,rbspace)
err = map(x,xrec) do x,xrec
  x - xrec
end

function _plot(ranks,path,name,trian,x)
  createpvd(ranks,path) do pvd
    for (xt,t) in x
      pvd[t] = createvtk(trian,path*"_$t.vtu",cellfields=[name=>xt])
    end
  end
end

μ = params[1]
times = get_times(fesolver)
sol = sols[1]

trial0 = trial(μ,times)
arr = PVector(sol,partition(trial0.gids))
uh = FEFunction(trial0,arr)

_plot(ranks,joinpath("plots/ptsol"),"u",Ω,uh)

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

  t0,tf,dt,θ = 0.,0.1,0.01,0.5

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
  sol = solve(ode_solver,feop,uh0,t0,tf)

  # for (uₕ,t) in sol
  #   writevtk(Ω,"poisson_transient_solution_$t"*".vtu",cellfields=["u"=>uₕ])
  # end
  createpvd(ranks,"gridap_plots/sol") do pvd
    for (uₕ,t) in sol
      pvd[t] = createvtk(Ω,"gridap_plots/sol_$t"*".vtu",cellfields=["u"=>uₕ])
    end
  end
end

with_debug() do distribute
  ranks = distribute(LinearIndices((4,)))
  results_t = main_gridap(ranks,params[1])
end

##############

model = CartesianDiscreteModel(domain,mesh_partition)

Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialPFESpace(test,g)
feop = AffineTransientPFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.1,0.01,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

function _project_recast(snap::PTArray,rb::RBSpace)
  mat = stack(snap.array)
  rb_proj = space_time_projection(mat,rb)
  array = recast(rb_proj,rb)
  PTArray(array)
end

w0 = get_free_dof_values(fesolver.uh0(params))
time_ndofs = num_time_dofs(fesolver)
uμt = PODESolution(fesolver,feop,params,w0,fesolver.t0,fesolver.tf)
println("Computing fe solution: time marching across $time_ndofs instants, for $nparams parameters")
stats = @timed begin
  _snaps = map(uμt) do (snap,n)
    copy(snap)
  end
end
_sols = Snapshots(_snaps)
_rbspace = reduced_basis(rbinfo,feop,_sols)
_x = _sols[1]
_xrec = _project_recast(_x,_rbspace)
_err = _x - _xrec
