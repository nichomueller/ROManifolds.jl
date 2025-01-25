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

# order = 1
# reffe = ReferenceFE(lagrangian,Float64,2)
# test = FESpace(Ωbg.trian,reffe;conformity=:H1)
# testact = FESpace(Ωact,reffe;conformity=:H1)

# test0 = FESpace(Ωbg.trian,reffe;conformity=:H1,constraint=:zeromean)

# V = OrderedFESpace(test)
# Vact = OrderedFESpace(testact)

# a = get_cell_dof_ids_with_zeros(Vact)

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

a = get_sparsity(f_act,f_act)

I_bg,J_bg, = DofMaps.bg_findnz(a.sparsity)
nrows_bg = DofMaps.num_bg_rows(a.sparsity)
ncols_bg = DofMaps.num_bg_cols(a.sparsity)
nrows = DofMaps.num_rows(a)
dsd2sd = DofMaps.get_d_sparse_dofs_to_full_dofs(a,I_bg,J_bg,nrows_bg,ncols_bg)
for (k,sdk) in enumerate(dsd2sd)
  if sdk > 0
    Ik_bg = fast_index(sdk,nrows_bg)
    Jk_bg = slow_index(sdk,nrows_bg)
    Ik = a.sparsity.bg_rows_to_act_rows[Ik_bg]
    Jk = a.sparsity.bg_cols_to_act_cols[Jk_bg]
    dsd2sd[k] = Ik+(Jk-1)*nrows
  end
end


# M = assemble_matrix(form,f.space,f.space)
# smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)

# test 4

reffe = ReferenceFE(lagrangian,Float64,2)
f = TProductFESpace(Ωbg,reffe;conformity=:H1)
f_act = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1,constraint=:zeromean)

smap = get_sparse_dof_map(f,f)
smap_act = get_sparse_dof_map(f_act,f_act)

M = assemble_matrix(form,f.space,f.space)
smarray = DofMapArray(M,smap_act.d_sparse_dofs_to_full_dofs)




a = A
I_bg,J_bg, = DofMaps.bg_findnz(a.sparsity)
nrows_bg = DofMaps.num_bg_rows(a.sparsity)
ncols_bg = DofMaps.num_bg_cols(a.sparsity)
nrows = DofMaps.num_rows(a)
dsd2sd = DofMaps.get_d_sparse_dofs_to_full_dofs(a,I_bg,J_bg,nrows_bg,ncols_bg)

nnz_sizes = DofMaps.univariate_nnz(a)
rows_no_comps = DofMaps.univariate_num_rows(a)
cols_no_comps = DofMaps.univariate_num_cols(a)
nrows_no_comps = prod(rows_no_comps)
ncols_no_comps = prod(cols_no_comps)
ncomps_row = Int(nrows_bg / nrows_no_comps)
ncomps_col = Int(ncols_bg / ncols_no_comps)
ncomps = ncomps_row*ncomps_col

i,j, = DofMaps.univariate_findnz(a)
d_to_nz_pairs = map((id,jd)->map(CartesianIndex,id,jd),i,j)

D = length(a.sparsities_1d)
cache = zeros(Int,D)
dsd2sd = zeros(Int,nnz_sizes...,ncomps)

k = 5863
I_node,I_comp = DofMaps._fast_and_slow_index(I_bg[k],nrows_no_comps)
J_node,J_comp = DofMaps._fast_and_slow_index(J_bg[k],ncols_no_comps)
comp = I_comp+(J_comp-1)*ncomps_row
rows_1d = _index_to_d_indices(I_node,rows_no_comps)
cols_1d = _index_to_d_indices(J_node,cols_no_comps)
_row_col_pair_to_nz_index!(cache,rows_1d,cols_1d,d_to_nz_pairs)

#
cell_odofs_ids = get_cell_dof_ids(f_act)
bg_f = f_act.space.space
bg_dofs_to_act_dofs = DofMaps.get_bg_dof_to_act_dof(bg_f)
bg_odof_to_act_odof = collect(1:length(bg_dofs_to_act_dofs))
k = testitem(cell_odofs_ids.maps)
for (bg_dof,act_dof) in enumerate(bg_dofs_to_act_dofs)
  if iszero(act_dof)
    bg_odof_to_act_odof[get_odof(k,bg_dof)] = 0
  end
end

for cell in 1:100
  k = cell_odofs_ids.maps[cell]
  bg_dofs = getindex!(bg_cache,bg_cell_ids,cell)
  act_dofs = getindex!(act_cache,act_cell_ids,cell)
  for (i,(bg_dof,act_dof)) in enumerate(zip(bg_dofs,act_dofs))
    if bg_mask[bg_dof]

      bg_odof_to_act_odof[k.pdof_to_dof[i]] = 0
    end
  end
  # act_dofs = getindex!(act_cache,act_cell_ids,cell)
end

using Gridap.Fields

function _find_constraint_cell_and_ldof(cellids,dof_mask)
  # cellmasks = lazy_map(Broadcasting(PosNegReindex(fv,dv)),cellids)
  cells = Vector{Int}(undef,length(dof_mask))
  ldofs = Vector{Int}(undef,length(dof_mask))
  cache = array_cache(cellids)
  for cell in 1:length(cellids)
    dofs = getindex!(cache,cellids,cell)

  end
end
