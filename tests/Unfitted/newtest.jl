modeltp = TProductModel(pmin,pmax,partition)
triantp = Triangulation(modeltp)
reffe = ReferenceFE(lagrangian,Float64,2)
V = TestFESpace(Ωact,reffe,conformity=:H1,dirichlet_tags="boundary")
Vtp = FESpace(triantp,reffe,conformity=:H1,dirichlet_tags="boundary")
_Vtp = FESpace(Ωbg,reffe,conformity=:H1,dirichlet_tags="boundary")
# n2n = get_node_to_parent_node(Ωact,Ωbg) this is actually wrong!

# cell_to_parent_cell = DofMaps.get_tface_to_mface(Ωact)

# cellids = get_cell_dof_ids(V)
# cellidstp = get_cell_dof_ids(Vtp)

dof_map = get_dof_map(bgmodel,V,fill(true,2,2))
dof_map_tp = get_dof_map(bgmodel,Vtp.space,fill(true,2,2))

# allinds = LinearIndices(dof_map)
# n2n = allinds[findall(!iszero,dof_map)]

α(x) = x[1]+x[2]
formvector(v) = ∫(α*v)dΩ
formmatrix(u,v) = ∫(α*u*v)dΩ

vector = assemble_vector(formvector,V)
vector_tp = assemble_vector(formvector,_Vtp)
matrix = assemble_matrix(formmatrix,V,V)
matrix_tp = assemble_matrix(formmatrix,_Vtp,_Vtp)

# norm(vector) == norm(vector_tp)

# tryvector = zeros(size(vector_tp))
# tryvector[n2n] = vector

# tryvector == vector_tp
# error_map = DofMapArray(vector_tp-tryvector,dof_map_tp)

DofMapArray(vector,dof_map) ≈ DofMapArray(vector_tp,dof_map_tp)

# inv_dof_map_tp = invert(dof_map_tp)
# inv_dof_map = invert(dof_map)

# # map1 ∘ map2
# function compose_maps(map1,map2)
#   @assert size(map1) == size(map2)
#   map12 = zeros(eltype(map1),size(map1))
#   for (i,m2i) in enumerate(map2)
#     iszero(m2i) && continue
#     map12[i] = map1[m2i]
#   end
#   return map12
# end

# map12 = compose_maps(dof_map,inv_dof_map_tp)
# v12 = vec(map12)

# DofMapArray(vector,v12) ≈ vector_tp

# map21 = compose_maps(dof_map_tp,inv_dof_map)
# v21 = vec(map21)
# v21 = v21[findall(!iszero,v21)]

# DofMapArray(vector_tp,v21) ≈ vector

dof_to_parent = DofMaps.get_dof_to_parent_dof_map(dof_map,dof_map_tp)
# sparsity = SparsityPattern(V,V)
# parent_sparsity = SparsityPattern(Vtp,Vtp)
# # s2parents = SparsityToTProductSparsity(sparsity,parent_sparsity,dof_to_parent,dof_to_parent)
# s2parents = SparsityToTProductSparsity(sparsity,parent_sparsity.sparsities_1d,dof_to_parent,dof_to_parent)
# sdof_map = get_sparse_dof_map(Vtp,Vtp,s2parents)
# sdof_map_parent = get_sparse_dof_map(Vtp,Vtp,parent_sparsity)

# A = DofMapArray(matrix,DofMaps.SparseDofMapIndexing(sdof_map))
# Atp = DofMapArray(matrix_tp,DofMaps.SparseDofMapIndexing(sdof_map_parent))

struct NewDofMapSparseMatrixCSC{Tv,Ti,A<:AbstractVector} <: AbstractSparseMatrix{Tv,Ti}
  matrix::SparseMatrixCSC{Tv,Ti}
  map_rows::A
  map_cols::A
end

Base.size(a::NewDofMapSparseMatrixCSC) = (length(a.map_rows),length(a.map_cols))

function Base.getindex(a::NewDofMapSparseMatrixCSC{Tv},i::Integer,j::Integer) where Tv
  i′ = a.map_rows[i]
  j′ = a.map_cols[j]
  if i′ != 0 && j′ != 0
    a.matrix[i′,j′]
  else
    zero(Tv)
  end
end

aa = NewDofMapSparseMatrixCSC(matrix,vec(dof_to_parent),vec(dof_to_parent))

aasp = sparse(aa)

aasp == matrix_tp

ii,jj,_ = findnz(aasp)

d2pd,l2pl = DofMaps._to_parent_indices(dof_to_parent)
I,J,V = findnz(matrix)
for k in eachindex(I)
  I[k] = l2pl[I[k]]
  J[k] = l2pl[J[k]]
end
I == ii
J == jj

# trial,test,sparsity = Vtp,Vtp,s2parents
# trian = get_triangulation(trial)
# model = get_background_model(trian)
# @assert model === get_background_model(get_triangulation(test))

# rows = get_dof_map(test)
# cols = get_dof_map(trial)
# unrows = get_univariate_dof_map(test)
# uncols = get_univariate_dof_map(trial)

# # osparsity,osparse_indices = get_sparse_dof_map(trian,rows,cols,unrows,uncols,sparsity)

# osparsity = order_sparsity(sparsity,(rows,unrows),(cols,uncols))
# I,J,V = findnz(osparsity)
# i,j,v = DofMaps.univariate_findnz(osparsity)
# sparse_indices = get_sparse_dof_map(osparsity,I,J,i,j)
# # osparse_indices = DofMaps.order_sparse_dof_map(sparse_indices,rows,cols)
# osparse_indices = DofMaps.order_sparse_dof_map(sparse_indices,dof_map,dof_map)

# A = DofMapArray(matrix,osparse_indices)

# for i in eachindex(A)
#   Ai = A[i]
#   Atpi = Atp[i]
#   if !iszero(A[i])
#     @assert Ai==Atpi "$i"
#   end
# end

# dof_to_parent_map = DofMap(
#   dof_to_parent,
#   dof_map_tp.dof_to_cell,
#   collect(LinearIndices(size(dof_to_parent))),
#   Fill(false,length(dof_map_tp.tface_to_mask)),
#   dof_map_tp.tface_to_mface)

# sparsity′ = order_sparsity(SparsityPattern(V,V),dof_to_parent_map,dof_to_parent_map)
