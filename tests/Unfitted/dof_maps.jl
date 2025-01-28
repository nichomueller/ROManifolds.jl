using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ROM
using ROM.DofMaps
using ROM.TProduct

R = 0.3
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
Ω = Triangulation(cutgeo,PHYSICAL_IN)

dΩ = Measure(Ω,2)
form(u,v) = ∫(u⋅v)dΩ

# test 1

reffe = ReferenceFE(lagrangian,Float64,2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1)
f_act = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1)

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f_act.space,f_act.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

norm(M) ≈ norm(smarray)

# test 2

reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1)
f_act = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1)

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f_act.space,f_act.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

norm(M) ≈ norm(smarray)

# test 3

using Gridap.CellData
using Gridap.FESpaces

reffe = ReferenceFE(lagrangian,Float64,2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1)
f_act = TProductFESpace(Ωbg,reffe;conformity=:H1,constraint=:zeromean)

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f_act.space,f_act.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

norm(M) ≈ norm(smarray)

# test 4

reffe = ReferenceFE(lagrangian,Float64,2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1)
f_act = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1,constraint=:zeromean)

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f_act.space,f_act.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

norm(M) ≈ norm(smarray)

# test 5

reffe = ReferenceFE(lagrangian,Float64,2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1,dirichlet_tags=[1,2,5])
f_act = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1,dirichlet_tags=[1,2,5])

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f_act.space,f_act.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

norm(M) ≈ norm(smarray)

# test 6

reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1,dirichlet_tags=[1,2,5])
f_act = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1,dirichlet_tags=[1,2,5])

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f_act.space,f_act.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

norm(M) ≈ norm(smarray)

# test 7

using Gridap.CellData
using Gridap.FESpaces

reffe = ReferenceFE(lagrangian,Float64,2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1,dirichlet_tags=[1,2,5])
f_act = TProductFESpace(Ωbg,reffe;conformity=:H1,constraint=:zeromean,dirichlet_tags=[1,2,5])

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f_act.space,f_act.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

norm(M) ≈ norm(smarray)

# test 8

reffe = ReferenceFE(lagrangian,Float64,2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1,dirichlet_tags=[1,2,5])
f_act = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1,constraint=:zeromean,dirichlet_tags=[1,2,5])

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f_act.space,f_act.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

norm(M) ≈ norm(smarray)
