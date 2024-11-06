using Gridap
using Test
using DrWatson
using Serialization

using GridapEmbedded

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

R  = 0.5
L  = 0.5*R
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(-L,L)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R
dp = pmax - pmin

n = 30
partition = (n,n)
bgmodel = TProductModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo3)
Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,ACTIVE_IN)
Ω_out = Triangulation(cutgeo,ACTIVE_OUT)

order = 1
degree = 2*order

dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)
Γ_in = EmbeddedBoundary(cutgeo)
nΓ_in = get_normal_vector(Γ_in)
dΓ_in = Measure(Γ_in,degree)

const γd = 10.0    # Nitsche coefficient
const h = dp[1]/n  # Mesh size according to the parameters of the background grid

ν_in(x,μ) = 1+exp(-x[1]/sum(μ))
ν_in(μ) = x->ν_in(x,μ)
νμ_in(μ) = ParamFunction(ν_in,μ)

const ν_out = 1e-6

f(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g0(x,μ) = 0
g0(μ) = x->g0(x,μ)
g0μ(μ) = ParamFunction(g0,μ)

stiffness(μ,u,v,dΩ_in,dΩ_out,dΓ_in) = ( ∫( νμ_in(μ)*∇(v)⋅∇(u) )dΩ_in + ∫( ν_out*∇(v)⋅∇(u) )dΩ_out
  + ∫( (γd/h)*v*u  - v*(nΓ_in⋅∇(u)) - (nΓ_in⋅∇(v))*u )dΓ_in)
rhs(μ,v,dΓ_in) = ∫( (γd/h)*v*fμ(μ) - (nΓ_in⋅∇(v))*fμ(μ) )dΓ_in
res(μ,u,v,dΩ_in,dΩ_out,dΓ_in) = rhs(μ,v,dΓ_in) - stiffness(μ,u,v,dΩ_in,dΩ_out,dΓ_in)

reffe = ReferenceFE(lagrangian,Float64,order)

trians = (Ω_in.trian,Ω_out.trian,Γ_in)
test = TestFESpace(Ω,reffe,conformity=:H1;dirichlet_tags=["boundary"])
trial = ParamTrialFESpace(test,g0μ)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,trians,trians)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
energy(du,v) = ∫(v*du)dΩ_in + ∫(∇(v)⋅∇(du))dΩ_in
state_reduction = TTSVDReduction(tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

#############################
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.Geometry
using ReducedOrderModels.TProduct

space = test.space
trian = Ω_in.trian
# get_dof_index_map(space,trian)

model = get_background_model(trian)
cell_dof_ids = get_cell_dof_ids(space)

trian_ids = trian.tface_to_mface

dof = get_fe_dof_basis(space)
T = TProduct.get_dof_type(dof)
order = get_polynomial_order(space)
comp_to_dofs = TProduct.get_comp_to_dofs(T,space,dof)

# get_dof_index_map(T,model,cell_dof_ids,trian_ids,order,comp_to_dofs)

Dc = 2
Ti = Int32
desc = get_cartesian_descriptor(model)

periodic = desc.isperiodic
ncells = desc.partition
ndofs = order .* ncells .+ 1 .- periodic

terms = TProduct._get_terms(first(get_polytopes(model)),fill(order,Dc))
cache_cell_dof_ids = array_cache(cell_dof_ids)

new_dof_ids = LinearIndices(ndofs)
dof_map = fill(zero(Ti),ndofs)

for (icell,cell) in enumerate(CartesianIndices(ncells))
  if icell ∈ trian_ids
    icell′ = findfirst(trian_ids.==icell)
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    new_dofs_range = map(i -> i:i+order,first_new_dof)
    new_dofs = view(new_dof_ids,new_dofs_range...)
    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell′)
    for (idof,dof) in enumerate(cell_dofs)
      t = terms[idof]
      dof < 0 && continue
      dof_map[new_dofs[t]] = dof
    end
  end
end
############################## WITH TPOD #####################################

bgmodel = CartesianDiscreteModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo3)
Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,ACTIVE_IN)
Ω_out = Triangulation(cutgeo,ACTIVE_OUT)

dΩ = Measure(Ω,degree)
dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)
Γ_in = EmbeddedBoundary(cutgeo)
nΓ_in = get_normal_vector(Γ_in)
dΓ_in = Measure(Γ_in,degree)

trians = (Ω_in,Ω_out,Γ_in)
test = TestFESpace(Ω,reffe,conformity=:H1;dirichlet_tags=["boundary"])
trial = ParamTrialFESpace(test,g0μ)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,trians,trians)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
state_reduction = Reduction(tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)
