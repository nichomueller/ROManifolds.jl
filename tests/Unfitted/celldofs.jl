modeltp = TProductModel(pmin,pmax,partition)
triantp = Triangulation(modeltp)
reffe = ReferenceFE(lagrangian,Float64,2)
V = TestFESpace(Ωact,reffe,conformity=:H1,dirichlet_tags="boundary")
Vtp = TProductFESpace(triantp,reffe,conformity=:H1,dirichlet_tags="boundary")
_Vtp = FESpace(Ωbg,reffe,conformity=:H1,dirichlet_tags="boundary")

dof_map = get_dof_map(bgmodel,V,fill(true,2,2))
dof_map_tp = get_dof_map(bgmodel,Vtp.space,fill(true,2,2))

α(x) = x[1]+x[2]
formvector(v) = ∫(α*v)dΩ
formmatrix(u,v) = ∫(α*u*v)dΩ

vector = assemble_vector(formvector,V)
vector_tp = assemble_vector(formvector,_Vtp)
matrix = assemble_matrix(formmatrix,V,V)
matrix_tp = assemble_matrix(formmatrix,_Vtp,_Vtp)

dof_to_parent = DofMaps.get_dof_to_parent_dof_map(dof_map,dof_map_tp)
sparsity = SparsityPattern(V,V)
parent_sparsity = SparsityPattern(Vtp,Vtp)
s2parents = SparsityToTProductSparsity(sparsity,parent_sparsity.sparsities_1d,dof_to_parent,dof_to_parent)
sdof_map_parent = get_sparse_dof_map(Vtp,Vtp,parent_sparsity)

# sdof_map = get_sparse_dof_map(Vtp,Vtp,s2parents)
trial,test,sparsity = Vtp,Vtp,s2parents
trian = get_triangulation(trial)
model = get_background_model(trian)
@assert model === get_background_model(get_triangulation(test))

rows = get_dof_map(test)
cols = get_dof_map(trial)
unrows = get_univariate_dof_map(test)
uncols = get_univariate_dof_map(trial)

# osparsity,osparse_indices = get_sparse_dof_map(trian,rows,cols,unrows,uncols,sparsity)
osparsity = order_sparsity(sparsity,(rows,unrows),(cols,uncols))
I,J,V = findnz(osparsity)
i,j,v = DofMaps.univariate_findnz(osparsity)
sparse_indices = get_sparse_dof_map(osparsity,I,J,i,j)
sparse_indices_wrong = get_sparse_dof_map(osparsity,I,J,i,j)
# osparse_indices = DofMaps.order_sparse_dof_map(sparse_indices,rows,cols)
osparse_indices = DofMaps.order_sparse_dof_map(sparse_indices,dof_map,dof_map)

nrows = size(matrix,1)
IJ = range_2d(vectorize(dof_map),vectorize(dof_map),nrows)

k = 76
dofk = sparse_indices[k]
IJ_sparse_k = sparse_indices[76]
Ik,Jk = fast_index(IJ_sparse_k,size(matrix,1)),slow_index(IJ_sparse_k,size(matrix,1))
IJ_sparse_wrong_k = sparse_indices_wrong[76]
Ik_wrong,Jk_wrong = fast_index(IJ_sparse_wrong_k,size(matrix_tp,1)),slow_index(IJ_sparse_wrong_k,size(matrix_tp,1))
dof_to_parent[Ik] == Ik_wrong
dof_to_parent[Ik_wrong] == Ik

parent_to_dof = copy(dof_to_parent)
fill!(parent_to_dof,zero(eltype(parent_to_dof)))
inz = findall(!iszero,dof_to_parent)
parent_to_dof[inz] = LinearIndices(parent_to_dof)[inz]

# parent2i = DofMaps.compose_maps(dof_map_tp,invert(dof_map)) why is this wrong??
# vector == DofMapArray(vector_tp,parent2i[inz])

# TRUE!
# vector == DofMapArray(vector_tp,parent_to_dof[inz])

nrows_wrong = size(matrix_tp,1)
IJ_wrong = range_2d(vectorize(dof_map_tp),vectorize(dof_map_tp),nrows_wrong)
odof_map = copy(sparse_indices)
fill!(odof_map,zero(eltype(odof_map)))
for (k,dofk) in enumerate(sparse_indices_wrong)
  if dofk > 0
    IJwrong = IJ_wrong[dofk]
    Iwrong = fast_index(IJwrong,nrows_wrong)
    Jwrong = slow_index(IJwrong,nrows_wrong)
    I = dof_to_parent[Iwrong]#parent2i[Iwrong]
    J = dof_to_parent[Jwrong]#parent2i[Jwrong]
    odof_map[k] = I + (J-1)*nrows
  end
end

A = DofMapArray(matrix,odof_map)




##### COMMENT ####
# # this works, but I need to restrict the codomain of sparse_map_Ωact
# sparsity_Ωbg = SparsityPattern(Vtp,Vtp)
# sparse_map_Ωbg = get_sparse_dof_map(Vtp,Vtp)
# dof_map_rows_Ωact = change_domain(Vtp.dof_map,Ωact)
# dof_map_cols_Ωact = dof_map_rows_Ωact
# sparsity_Ωact = change_domain(sparsity_Ωbg,dof_map_rows_Ωact,dof_map_cols_Ωact)
# sparse_map_Ωact = get_sparse_dof_map(Vtp,Vtp,sparsity_Ωact)
# Atp = DofMapArray(matrix_tp,DofMaps.SparseDofMapIndexing(sparse_map_Ωact))

# DOES THIS WORK?
dof_to_parent = DofMaps.get_dof_to_parent_dof_map(dof_map,dof_map_tp)
sparsity = SparsityPattern(V,V)
parent_sparsity = SparsityPattern(Vtp,Vtp)
s2parents = SparsityToTProductSparsity(sparsity,parent_sparsity.sparsities_1d,dof_to_parent,dof_to_parent)
sdof_map_parent = get_sparse_dof_map(Vtp,Vtp,parent_sparsity)

# sdof_map = get_sparse_dof_map(Vtp,Vtp,s2parents)
trial,test,sparsity = Vtp,Vtp,s2parents
trian = get_triangulation(trial)
model = get_background_model(trian)
@assert model === get_background_model(get_triangulation(test))

rows = get_dof_map(test)
cols = get_dof_map(trial)
unrows = get_univariate_dof_map(test)
uncols = get_univariate_dof_map(trial)

# osparsity,osparse_indices = get_sparse_dof_map(trian,rows,cols,unrows,uncols,sparsity)
osparsity = order_sparsity(sparsity,(rows,unrows),(cols,uncols))
I,J,V = findnz(osparsity)
i,j,v = DofMaps.univariate_findnz(osparsity)
sparse_indices = get_sparse_dof_map(osparsity,I,J,i,j)
osparse_indices = DofMaps.new_order_sparse_dof_map(
  sparse_indices,dof_to_parent,dof_map_tp,dof_map_tp,size(matrix,1),size(matrix_tp,1))

DofMapArray(matrix,osparse_indices) ≈ Atp
