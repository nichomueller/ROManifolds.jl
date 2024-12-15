using Gridap
using GridapEmbedded
using Test
using Gridap
using GridapEmbedded
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.CellData
using Gridap.Geometry
using DrWatson
using Serialization
using SparseArrays

using ReducedOrderModels
using ReducedOrderModels.DofMaps
using ReducedOrderModels.TProduct

u(x) = x[1] - x[2]
f(x) = -Δ(u)(x)
ud(x) = u(x)

R = 0.5
L = 0.8*(2*R)
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(L,0.0)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin
const h = dp[1]/n

cutgeo = cut(bgmodel,geo3)

# Setup integration meshes
Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)

# Setup normal vectors
n_Γ = get_normal_vector(Γ)
n_Γg = get_normal_vector(Γg)

# Setup Lebesgue measures
order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

# Setup FESpace
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(Ωact,reffe,conformity=:H1)
U = TrialFESpace(V)

# Weak form Nitsche + ghost penalty (CutFEM paper Sect. 6.1)
const γd = 10.0
const γg = 0.1

a(u,v) =
  ∫( ∇(v)⋅∇(u) ) * dΩ +
  ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u ) * dΓ +
  ∫( (γg*h)*jump(n_Γg⋅∇(v))*jump(n_Γg⋅∇(u)) ) * dΓg

l(v) =
  ∫( v*f ) * dΩ +
  ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud ) * dΓ

# FE problem
op = AffineFEOperator(a,l,U,V)
uh = solve(op)

dof_map = get_dof_map(bgmodel,V)
vals = DofMapArray(uh.free_values,dof_map)

dof_map = get_dof_map(bgmodel,V)
A = copy(op.op.matrix)
fill!(A.nzval,0.0)

modeltp = TProductModel(pmin,pmax,partition)
triantp = Triangulation(modeltp)
Vtp = FESpace(triantp,reffe,conformity=:H1)

cellids = get_cell_dof_ids(V)
cellids_tp = get_cell_dof_ids(Vtp)

for (i,idi) in enumerate(Ωact.tface_to_mface)
  @assert cellids[i] == cellids_tp[idi] "$i"
end

sparsity_1d = map(SparsityPattern,Vtp.spaces_1d,Vtp.spaces_1d)

sparsity = TProductSparsity(SparsityPattern(A),sparsity_1d)

rows = copy(dof_map)
cols = copy(dof_map)
unrows = TProduct.get_univariate_dof_map(Vtp)
uncols = TProduct.get_univariate_dof_map(Vtp)

# osparsity = order_sparsity(sparsity,(rows,unrows),(cols,uncols))
i,j = rows,cols
# matrix = A[i,j]
I′ = DofMaps.vectorize(i)
J′ = DofMaps.vectorize(j)

fill!(A.nzval,one(Float64))
k = findfirst(iszero,A)
fill!(A.nzval,zero(Float64))
iz,jz = Tuple(k)

for (ii,i) in enumerate(I′)
  if i == 0
    I′[ii] = iz
  end
end

for (ij,j) in enumerate(J′)
  if j == 0
    J′[ij] = jz
  end
end

sparsity_ok = SparsityPattern(Vtp,Vtp)
A_ok = sparsity_ok.sparsity.matrix
I_ok,J_ok,V_ok = findnz(A_ok)
I,J,V = findnz(A)

IJ_ok = eachrow(hcat(I_ok,J_ok))
IJ = eachrow(hcat(I,J))

findall(isnothing,indexin(IJ_ok,IJ))
findall(isnothing,indexin(IJ,IJ_ok))

CIAO
# I,J,V = findnz(osparsity)
# i,j,v = univariate_findnz(osparsity)
# sparse_indices = get_sparse_dof_map(osparsity,I,J,i,j)
# osparse_indices = order_sparse_dof_map(sparse_indices,rows,cols)

# A = op.op.matrix

# dof_map = get_dof_map(VV)
# dof_map_Ω = change_domain(dof_map,Ω)

# j = 31*30+1
# i = dof_map_Ω
# free_j = i.free_vals_box[j]
# dof_j = i.indices[free_j]

###################################THIS WORKS##################################

using Gridap
using GridapEmbedded
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.CellData
using DrWatson
using Serialization

using ReducedOrderModels
using ReducedOrderModels.DofMaps

const L = 1
const R  = 0.1
const n = 30

const domain = (0,L,0,L)
const partition = (n,n)

p1 = Point(0.3,0.5)
p2 = Point(0.7,0.5)
geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = !union(geo1,geo2)

bgmodel = TProductModel(domain,partition)
cutgeo = cut(bgmodel,geo3)

Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γd = EmbeddedBoundary(cutgeo)
Γn = BoundaryTriangulation(bgmodel;tags="boundary")

order = 2

trian_res = (Ω_in,Γd,Γn)
trian_jac = (Ω_in,Ω_out,Γd)
domains = FEDomains(trian_res,trian_jac)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1)

dof_map = get_dof_map(test_u)

dof_map_in = change_domain(dof_map,Ω_in)
dof_map_out = change_domain(dof_map,Ω_out)
dof_map_Γd = change_domain(dof_map,Γd)
dof_map_Γn = change_domain(dof_map,Γn)

Ω_act = Triangulation(cutgeo,ACTIVE)
V = TestFESpace(Ω_act,reffe_u;conformity=:H1)
dof_map_try = get_dof_map(bgmodel.model,V)

################################################################################

# understand how cell_dof_ids generation works

using Gridap.Arrays
using Gridap.ReferenceFEs

trian = Ωact
model = get_active_model(trian)
basis,reffe_args,reffe_kwargs = reffe
cell_reffe = ReferenceFEs.ReferenceFE(model,basis,reffe_args...;reffe_kwargs...)
# FESpace(model,cell_reffe;trian,reffe_kwargs...)
conformity=:H1
trian = Triangulation(model)
labels = get_face_labeling(model)
dirichlet_tags=Int[]
dirichlet_masks=nothing
constraint=nothing
vector_type=nothing
conf = Conformity(testitem(cell_reffe),conformity)
FESpaces._use_clagrangian(trian,cell_reffe,conf)
num_vertices(model) == num_nodes(model)
ctype_reffe, cell_ctype = compress_cell_data(cell_reffe)
prebasis = get_prebasis(first(ctype_reffe))
T = return_type(prebasis)
# Next line assumes linear grids
node_to_tag = Geometry.get_face_tag_index(labels,dirichlet_tags,0)
_vector_type = vector_type === nothing ? Vector{Float64} : vector_type
tag_to_mask = dirichlet_masks === nothing ? fill(FESpaces._default_mask(T),length(dirichlet_tags)) : dirichlet_masks
vector_type = Vector{FESpaces._dof_type(T)}
grid = trian
node_to_tag = fill(Int8(UNSET),num_nodes(grid))
tag_to_mask = fill(FESpaces._default_mask(T),0)
z = zero(T)
glue, dirichlet_dof_tag = FESpaces._generate_node_to_dof_glue_component_major(
  z,node_to_tag,tag_to_mask)
cell_reffe = FESpaces._generate_cell_reffe_clagrangian(z,grid)
# cell_dofs_ids = FESpaces._generate_cell_dofs_clagrangian(z,grid,glue,cell_reffe)
ctype_to_reffe, cell_to_ctype = compress_cell_data(cell_reffe)
cell_to_nodes = get_cell_node_ids(grid)


################################################################################

parent = get_grid(triantp.trian)
cell_to_parent_cell = Ωact.tface_to_mface

Dc = num_cell_dims(parent)
Dp = num_point_dims(parent)

parent_cell_to_parent_nodes = get_cell_node_ids(parent)
nparent_nodes = num_nodes(parent)
parent_node_to_coords = get_node_coordinates(parent)

node_to_parent_node, parent_node_to_node = Geometry._find_active_nodes(
  parent_cell_to_parent_nodes,cell_to_parent_cell,nparent_nodes)

cell_to_nodes = Geometry._renumber_cell_nodes(
  parent_cell_to_parent_nodes,parent_node_to_node,cell_to_parent_cell)

cellids = get_cell_dof_ids(V)
cellids_tp = get_cell_dof_ids(Vtp)

i = 1
itp = cell_to_parent_cell[i]

for (i,itp) in enumerate(Ωact.tface_to_mface)
  idsi = cellids[i]
  idstp = cellids_tp[itp]
  for k in eachindex(idsi)
    @assert node_to_parent_node[idsi[k]] == idstp[k] "$i, $k"
  end
end

node_to_parent_node[cellids[i][1]] == cellids_tp[itp][1]
node_to_parent_node[cellids[i][2]] == cellids_tp[itp][2]
node_to_parent_node[cellids[i][3]] == cellids_tp[itp][3]
node_to_parent_node[cellids[i][4]] == cellids_tp[itp][4]

A = SparsityPattern(V,V).matrix
A.nzval .= rand(length(A.nzval))
Atp = SparsityPattern(Vtp,Vtp).sparsity.matrix
B = copy(Atp)
fill!(B,0.0)
for col in axes(A,2)
  parent_col = node_to_parent_node[col]
  for k in SparseArrays.getcolptr(A)[col] : (SparseArrays.getcolptr(A)[col+1]-1)
    row = rowvals(A)[k]
    parent_row = node_to_parent_node[row]
    B[parent_row,parent_col] = A[row,col]
  end
end
Ib,Jb,Vb = findnz(B)
