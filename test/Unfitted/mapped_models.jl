using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.CellData
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ODEs
using ROManifolds
using ROManifolds.ParamDataStructures
using ROManifolds.ParamSteady
using Test
using FillArrays

# domain = (0,1,0,1)
# partition = (5,5)
# model = CartesianDiscreteModel(domain,partition)

# φ(x) = VectorValue(x[2],3*x[1])
# φt(x) = VectorValue(3*x[2],x[1])
# mmodel = MappedDiscreteModel(model,φ)

# Ω = Triangulation(model)
# Γ = BoundaryTriangulation(model,tags=8)
# Ωm = Triangulation(mmodel)
# Γm = BoundaryTriangulation(mmodel,tags=8)

# dΩ = Measure(Ω,4)
# dΓ = Measure(Γ,4)
# dΩm = Measure(Ωm,4)
# dΓm = Measure(Γm,4)

# g(x) = x[1]+x[2]

# reffe = ReferenceFE(lagrangian,Float64,2)
# V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
# U = TrialFESpace(V,g)
# Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
# Um = TrialFESpace(Vm,g)

# ν(x) = exp(-x[1])
# f(x) = x[2]

# atrian(u,v,dΩ) = ∫(ν*∇(v)⋅∇(u))dΩ
# btrian(v,dΩ,dΓ) = ∫(f*v)dΩ + ∫(f*v)dΓ

# a(u,v) = atrian(u,v,dΩ)
# b(v) = btrian(v,dΩ,dΓ)
# am(u,v) = atrian(u,v,dΩm)
# bm(v) = btrian(v,dΩm,dΓm)

# op = AffineFEOperator(a,b,U,V)
# opm = AffineFEOperator(am,bm,Um,Vm)

# uh = solve(op)
# uhm = solve(opm)

# v = get_fe_basis(V)
# u = get_trial_fe_basis(V)
# jcell = a(u,v)[Ω]

# vm = get_fe_basis(Vm)
# um = get_trial_fe_basis(Vm)
# jcellm = am(um,vm)[Ωm]

# detJφ = 3
# Jφt = CellField(∇(φ),Ω)
# νm = ν∘φ
# mappedj = (∫( νm*(inv(Jφt)⋅∇(v)) ⋅ (inv(Jφt)⋅∇(u))*detJφ )dΩ)[Ω]

# ncells = num_cells(Ω)
# compare = lazy_map(≈,jcellm,mappedj)
# @assert sum(compare) == ncells


#

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

pspace = ParamSpace((3.0,4.0))

μ = Realization([[3.0],[4.0]])
ϕ(μ) = x->VectorValue(x[2],μ[1]*x[1])
ϕμ(μ) = parameterize(ϕ,μ)
mmodel = MappedDiscreteModel(model,ϕμ(μ))

Ωm = Triangulation(mmodel)
Γm = BoundaryTriangulation(mmodel,tags=8)

dΩm = Measure(Ωm,4)
dΓm = Measure(Γm,4)

g(μ) = x->x[1]+μ[1]*x[2]
gμ(μ) = parameterize(g,μ)

reffe = ReferenceFE(lagrangian,Float64,2)
Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
Um = ParamTrialFESpace(Vm,gμ)

Umμ = Um(μ)

ν(μ) = x->x[1]+μ[1]*x[2]
νμ(μ) = parameterize(ν,μ)
f(μ) = x->x[1]+μ[1]*x[2]
fμ(μ) = parameterize(f,μ)

am(μ,u,v) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩm
bm(μ,u,v) = ∫(fμ(μ)*v)dΩm + ∫(fμ(μ)*v)dΓm

opm = LinearParamFEOperator(bm,am,pspace,Um,Vm)
# uhm = solve(opm)

u = zero(Um(μ))
x = get_free_dof_values(u)
op = get_algebraic_operator(opm)
nlop = ParamNonlinearOperator(op,μ)
solve!(x,LUSolver(),nlop)

v = get_fe_basis(Vm)
u = get_trial_fe_basis(Vm)
νμ(μ)*∇(v)⋅∇(u)

cell_∇a = lazy_map(Broadcasting(∇),get_data(v))
cell_map = get_cell_map(get_triangulation(v))
# lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
cell_Jt = lazy_map(∇,cell_map)
cell_invJt = lazy_map(Operation(pinvJt),cell_Jt)
k = Broadcasting(Operation(⋅))
args = cell_invJt,cell_∇a
fi = map(testitem,args)
T = return_type(k,fi...)

∇(v)⋅∇(u)
