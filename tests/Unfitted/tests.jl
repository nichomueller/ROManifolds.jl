using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.CellData

using ReducedOrderModels
using ReducedOrderModels.DofMaps
using ReducedOrderModels.TProduct

using SparseArrays

n = 20
partition = (n,n)
model = CartesianDiscreteModel((0,1,0,1),partition)

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe,conformity=:H1;dirichlet_tags=["boundary"])

trian = Triangulation(model)

i = get_dof_map(model,test)
s = SparsityPattern(test,test)

trianv = view(trian,[1,10,100])
iv = change_domain(i,trianv)
sv = change_domain(s,iv,iv)

cell_dof_ids = get_cell_dof_ids(test)

trial = test
imap = i

model1d = CartesianDiscreteModel((0,1),(n,))
test1d = TestFESpace(model1d,reffe,conformity=:H1;dirichlet_tags=["boundary"])
trial1d = test1d
imap1d = [get_dof_map(model1d,test1d),get_dof_map(model1d,test1d)]

sparsity = SparsityPattern(trial,test)
sparsities_1d = [SparsityPattern(trial1d,test1d),SparsityPattern(trial1d,test1d)]
sparsity = TProductSparsityPattern(sparsity,sparsities_1d)
psparsity = order_sparsity(sparsity,(imap,imap1d),(imap,imap1d))
I,J,_ = findnz(psparsity)
i,j,_ = DofMaps.univariate_findnz(psparsity)
sparse_indices = get_sparse_dof_map(psparsity,I,J,i,j)
osparse_indices = DofMaps.order_sparse_dof_map(sparse_indices,imap,imap,num_free_dofs(test))
ofull_indices = DofMaps.to_nz_index(osparse_indices,sparsity)

# try multi variate
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test = TestFESpace(model,reffe,conformity=:H1;dirichlet_tags=["boundary"])

i = get_dof_map(model,test)
s = SparsityPattern(test,test)

iv = change_domain(i,trianv)
sv = change_domain(s,iv,iv)

trial = test
imap = i

reffe1d = ReferenceFE(lagrangian,Float64,order)
test1d = TestFESpace(model1d,reffe1d,conformity=:H1;dirichlet_tags=["boundary"])
trial1d = test1d
imap1d = [get_dof_map(model1d,test1d),get_dof_map(model1d,test1d)]

sparsity = SparsityPattern(trial,test)
sparsities_1d = [SparsityPattern(trial1d,test1d),SparsityPattern(trial1d,test1d)]
sparsity = TProductSparsityPattern(sparsity,sparsities_1d)
# psparsity = order_sparsity(sparsity,(imap,imap1d),(imap,imap1d))

# imap′ = DofMaps.get_component(imap)
osparsity,osparse_indices = get_sparse_dof_map(trian,imap,imap,imap1d,imap1d,sparsity)
ofull_indices = to_nz_index(osparse_indices,sparsity)
smap = SparseDofMap(ofull_indices,osparse_indices,osparsity)

# mixed case
reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe,conformity=:H1;dirichlet_tags=["boundary"])
imap′ = get_dof_map(model,test)
sparsity = SparsityPattern(trial,test)
sparsity = TProductSparsityPattern(sparsity,sparsities_1d)
osparsity,osparse_indices = get_sparse_dof_map(trian,imap′,imap,imap1d,imap1d,sparsity)
ofull_indices = to_nz_index(osparse_indices,sparsity)
smap = SparseDofMap(ofull_indices,osparse_indices,osparsity)
sssmap = DofMaps.SparseDofMapIndexing(smap)
sssmap1 = sssmap[:,:,2]
smap1 = sparsity.sparsity.matrix[sssmap1]
