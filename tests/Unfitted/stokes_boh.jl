using Gridap
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.ReferenceFEs
using ROM
using ROM.Utils
using ROM.DofMaps
using ROM.TProduct

pmin = Point(0,0)
pmax = Point(1,1)
n = 3
partition = (n,n)
model = TProductModel(pmin,pmax,partition)
trian = Triangulation(model)

reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
tpspace = TProductFESpace(trian,reffe;conformity=:H1,dirichlet_tags=[1,2,5])

get_dof_map(tpspace)
get_sparse_dof_map(tpspace,tpspace)

reffe0 = ReferenceFE(lagrangian,Float64,2)
tpspace0 = TProductFESpace(trian,reffe0;conformity=:H1,dirichlet_tags=[1,2,5],constraint=:zeromean)

get_dof_map(tpspace0)

CIAO
# Ω = trian.trian
# dΩ = Measure(Ω,2)
# reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
# space = FESpace(model.model,reffe;conformity=:H1,dirichlet_tags=[1,2,5])
# space′ = OrderedFESpace(space)

# form(u,v) = ∫(u⋅v)dΩ
# A = assemble_matrix(form,space,space)
# A′ = assemble_matrix(form,space′,space′)

# dmap = get_dof_map(model.model,space,[false false
#   true false])
# A11 = A[dmap[:,:,1][:],dmap[:,:,1][:]]
# A21 = A[dmap[:,:,2][:],dmap[:,:,1][:]]
# A12 = A[dmap[:,:,1][:],dmap[:,:,2][:]]
# A22 = A[dmap[:,:,2][:],dmap[:,:,2][:]]
# Matrix([A11 A12
#   A21 A22]) ≈ A′

# # linear system

# j(u,v) = ∫(u⋅v)dΩ + ∫(∇(u)⊙∇(v))dΩ
# l(v) = ∫(Point(1,1)⋅v)dΩ

# op = AffineFEOperator(j,l,space,space)
# op′ = AffineFEOperator(j,l,space′,space′)

# uh = solve(op)
# u = uh.free_values
# uh′ = solve(op′)
# u′ = uh′.free_values

# u1 = u[dmap[:,:,1][:]]
# u2 = u[dmap[:,:,2][:]]

# vcat(u1,u2) ≈ u′

# writevtk(Ω,"boh",cellfields=["err"=>uh-uh′])

using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ROM

R = 0.2
pmin = Point(0,0)
pmax = Point(1,1)
n = 10
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = TProductModel(pmin,pmax,partition)
cutgeo = cut(bgmodel,geo2)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
test = FESpace(Ωbg.trian,reffe;conformity=:H1)
testact = FESpace(Ωact,reffe;conformity=:H1)

test0 = FESpace(Ωbg.trian,reffe;conformity=:H1,constraint=:zeromean)
