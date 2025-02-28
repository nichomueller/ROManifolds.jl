using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using ROManifolds
using ROManifolds.ParamDataStructures
using Test
using FillArrays

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

φ(x) = VectorValue(x[2],3*x[1])
φt(x) = VectorValue(3*x[2],x[1])
mmodel = MappedDiscreteModel(model,φ)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model,tags=8)
Ωm = Triangulation(mmodel)
Γm = BoundaryTriangulation(mmodel,tags=8)

dΩ = Measure(Ω,4)
dΓ = Measure(Γ,4)
dΩm = Measure(Ωm,4)
dΓm = Measure(Γm,4)

g(x) = x[1]+x[2]

reffe = ReferenceFE(lagrangian,Float64,2)
V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
U = TrialFESpace(V,g)
Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
Um = TrialFESpace(Vm,g)

ν(x) = exp(-x[1])
f(x) = x[2]

atrian(u,v,dΩ) = ∫(ν*∇(v)⋅∇(u))dΩ
btrian(v,dΩ,dΓ) = ∫(f*v)dΩ + ∫(f*v)dΓ

a(u,v) = atrian(u,v,dΩ)
b(v) = btrian(v,dΩ,dΓ)
am(u,v) = atrian(u,v,dΩm)
bm(v) = btrian(v,dΩm,dΓm)

op = AffineFEOperator(a,b,U,V)
opm = AffineFEOperator(am,bm,Um,Vm)

uh = solve(op)
uhm = solve(opm)

v = get_fe_basis(V)
u = get_trial_fe_basis(V)
jcell = a(u,v)[Ω]

vm = get_fe_basis(Vm)
um = get_trial_fe_basis(Vm)
jcellm = am(um,vm)[Ωm]

detJφ = 3
Jφt = CellField(∇(φ),Ω)
νm = ν∘φ
mappedj = (∫( νm*(inv(Jφt)⋅∇(v)) ⋅ (inv(Jφt)⋅∇(u))*detJφ )dΩ)[Ω]

ncells = num_cells(Ω)
compare = lazy_map(≈,jcellm,mappedj)
@assert sum(compare) == ncells


#


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
using Gridap.FESpaces
using Gridap.ODEs

objects = gμ(μ)
space = allocate_space(Um,μ)
dir_values = get_dirichlet_dof_values(space)
dir_values_scratch = zero_dirichlet_values(space)
# dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
dirichlet_dof_to_tag = get_dirichlet_dof_tag(space)
cell_vals = FESpaces._cell_vals(space,objects)
FESpaces.gather_dirichlet_values!(dir_values_scratch,space,cell_vals)
free_values = zero_free_values(space)

using Gridap.CellData
s = get_fe_dof_basis(space)
trian = get_triangulation(s)
cf = CellField(objects,trian,DomainStyle(s))
b = change_domain(cf,s.domain_style)
# lazy_map(evaluate,get_data(s),get_data(b))
# cache = return_cache(get_data(s)[1],get_data(b)[1])
# ye = evaluate!(cache,get_data(s)[1],get_data(b)[1])
field = get_data(b)[1]
basis = get_data(s)[1]
c = return_cache(field,basis.nodes)
vals = evaluate!(c,field,basis.nodes)
ndofs = length(basis.dof_to_node)
r = _lagr_dof_cache(vals,ndofs)




ν(x) = exp(-x[1])
f(x) = x[2]

a(u,v) = ∫(ν*∇(v)⋅∇(u))dΩm
b(v) = ∫(f*v)dΩm + ∫(f*v)dΓm

op = AffineFEOperator(a,b,Um,Vm)
uhm = solve(opm)


cell_map = get_cell_map(get_triangulation(cf))
cell_field_phys = get_data(cf)
cell_field_ref = lazy_map(Broadcasting(∘),cell_field_phys,cell_map)
return_value(Broadcasting(∘),cell_field_phys[1],cell_map[1])

Ω = Triangulation(model)
cf′ = CellField(x->x[1],Ω)
cell_map′ = get_cell_map(get_triangulation(cf′))
cell_field_phys′ = get_data(cf′)
lazy_map(Broadcasting(∘),cell_field_phys′,cell_map′)
c′ = return_cache(Broadcasting(∘),cell_field_phys′[1],cell_map′[1])
evaluate!(c′,Broadcasting(∘),cell_field_phys′[1],cell_map′[1])
