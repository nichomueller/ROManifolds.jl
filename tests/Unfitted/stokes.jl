using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

const L = 1
const R  = 0.1
const n = 30

const domain = (0,L,0,L)
const partition = (n,n)

p1 = Point(0.3,0.5)
p2 = Point(0.7,0.5)
geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = union(geo1,geo2)

bgmodel = TProductModel(domain,partition)
labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"wall",collect(1:6))
add_tag_from_tags!(labels,"inlet",[7])
add_tag_from_tags!(labels,"outlet",[8])

cutgeo = cut(bgmodel,geo3)

Ω = Triangulation(bgmodel)
Γ_inlet = BoundaryTriangulation(bgmodel;tags="inlet")
Γ_outlet = BoundaryTriangulation(bgmodel;tags="outlet")
Ω_in = Triangulation(cutgeo,PHYSICAL_OUT) # fix this
Ω_out = Triangulation(cutgeo,PHYSICAL_IN) # fix this
Γ_in = EmbeddedBoundary(cutgeo)

nΓ_inlet = get_normal_vector(Γ_inlet)
nΓ_outlet = get_normal_vector(Γ_outlet)
nΓ_in = get_normal_vector(Γ_in)

order = 2
degree = 2*order

dΩ = Measure(Ω,degree)
dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)

dΓ_in = Measure(Γ_in,degree)
dΓ_inlet = Measure(Γ_inlet,degree)
dΓ_outlet = Measure(Γ_outlet,degree)

ν(x,μ) = μ[1]*exp(-μ[2])
ν(μ) = x->ν(x,μ)
νμ(μ) = ParamFunction(ν,μ)

g_in(x,μ) = VectorValue(-x[2]*(1.0-x[2])*abs(μ[3]*sin(μ[2])/10),0.0)
g_in(μ) = x->g_in(x,μ)
gμ_in(μ) = ParamFunction(g_in,μ)
g_0(x,μ) = VectorValue(0.0,0.0)
g_0(μ) = x->g_0(x,μ)
gμ_0(μ) = ParamFunction(g_0,μ)

ν0 = 1e-3   # Artificial viscosity
γd = 10    # Nitsche coefficient (Dirichlet)
γn = 10    # Nitsche coefficient (Neumann)
h = L/n     # Mesh size

function nitsche_jac(μ,u,v,dΓ,nΓ)
  ∫( (γd/h)*v⋅u  - νμ(μ)*v⋅(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))⋅u )dΓ
end
function nitsche_res(μ,v,dΓ,nΓ)
  ∫( gμ_in(μ) ⋅ ((γd/h)*v  - νμ(μ)*(nΓ⋅∇(v))) )dΓ
end
function extension(μ,u,v,dΩ_out)
  ∫( ν0*∇(v)⊙∇(u) )dΩ_out
end

function a11(μ,u,v,dΩ_in,dΓ_in,dΓ_inlet)
  (
    ∫( νμ(μ)*∇(v)⊙∇(u) )dΩ_in
    + nitsche_jac(μ,u,v,dΓ_in,nΓ_in)
    + nitsche_jac(μ,u,v,dΓ_inlet,nΓ_inlet)
  )
end

function stiffness(μ,(u,p),(v,q),dΩ_in,dΓ_in,dΓ_inlet)
  a11(μ,u,v,dΩ_in,dΓ_in,dΓ_inlet) - ∫(p*(∇⋅(v)))dΩ_in + ∫(q*(∇⋅(u)))dΩ_in
end

function jac(μ,(u,p),(v,q),dΩ_in,dΩ_out,dΓ_in,dΓ_inlet)
  stiffness(μ,(u,p),(v,q),dΩ_in,dΓ_in,dΓ_inlet) + extension(μ,u,v,dΩ_out)
end

function res(μ,(u,p),(v,q),dΩ_in,dΩ_out,dΓ_in,dΓ_inlet)
  nitsche_res(μ,v,dΓ_inlet,nΓ_inlet) - ∫( νμ(μ)*∇(v)⊙∇(u) )dΩ_in
end

trian_res = (Ω_in,Ω_out,Γ_in,Γ_inlet)
trian_jac = (Ω_in,Ω_out,Γ_in,Γ_inlet)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags="wall")
trial_u = ParamTrialFESpace(test_u,gμ_0)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(res,jac,pspace,trial,test,trian_res,trian_jac)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=30,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=10,nparams_jac=10)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

r = realization(feop)
fesnaps,festats = solution_snapshots(rbsolver,feop,r)
