"""
    abstract type SparsityPattern end

Type used to represent the sparsity pattern of a sparse matrix, usually the
jacobian in a FE problem.

Subtypes:
- `SparsityCSC`
- `TProductSparsity`
"""
abstract type SparsityPattern end

function get_sparsity(U::FESpace,V::FESpace,trian=_get_common_domain(U,V))
  SparsityPattern(U,V,trian)
end

function SparsityPattern(U::FESpace,V::FESpace,trian=_get_common_domain(U,V))
  a = SparseMatrixAssembler(U,V)
  m1 = nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  cellidsrows = get_cell_dof_ids(V,trian)
  cellidscols = get_cell_dof_ids(U,trian)
  trivial_symbolic_loop_matrix!(m1,cellidsrows,cellidscols)
  m2 = nz_allocation(m1)
  trivial_symbolic_loop_matrix!(m2,cellidsrows,cellidscols)
  m3 = create_from_nz(m2)
  SparsityPattern(m3)
end

function SparsityPattern(a::AbstractSparseMatrix)
  @abstractmethod
end

get_background_matrix(a::SparsityPattern) = @abstractmethod
get_background_sparsity(a::SparsityPattern) = SparsityPattern(get_background_matrix(a))

num_rows(a::SparsityPattern) = size(get_background_matrix(a),1)
num_cols(a::SparsityPattern) = size(get_background_matrix(a),2)
SparseArrays.findnz(a::SparsityPattern) = findnz(get_background_matrix(a))
SparseArrays.nnz(a::SparsityPattern) = nnz(get_background_matrix(a))
SparseArrays.nonzeros(a::SparsityPattern) = nonzeros(get_background_matrix(a))
SparseArrays.nzrange(a::SparsityPattern,row::Integer) = nzrange(get_background_matrix(a),row)
SparseArrays.rowvals(a::SparsityPattern) = rowvals(get_background_matrix(a))
SparseArrays.getcolptr(a::SparsityPattern) = SparseArrays.getcolptr(get_background_matrix(a))
Algebra.nz_index(a::SparsityPattern,row::Integer,col::Integer) = nz_index(get_background_matrix(a),row,col)

function to_nz_index(i::AbstractArray,a::SparsityPattern)
  to_nz_index(i,get_background_matrix(i))
end

function get_sparse_dof_map(a::SparsityPattern,U::FESpace,V::FESpace,args...)
  TrivialSparseMatrixDofMap(a)
end

recast(v::AbstractVector,A::AbstractSparseMatrix) = @abstractmethod
recast(v::AbstractVector,A::SparseMatrixCSC) = SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,v)
recast(v::AbstractVector,A::SparseMatrixCSR{Bi}) where Bi = SparseMatrixCSR{Bi}(A.m,A.n,A.rowptr,A.colval,v)
recast(A::AbstractArray,a::SparsityPattern) = recast(A,get_background_matrix(a))

"""
    struct SparsityCSC{Tv,Ti} <: SparsityPattern
      matrix::SparseMatrixCSC{Tv,Ti}
    end
"""
struct SparsityCSC{Tv,Ti} <: SparsityPattern
  matrix::SparseMatrixCSC{Tv,Ti}
end

function SparsityPattern(a::SparseMatrixCSC)
  SparsityCSC(a)
end

get_background_matrix(a::SparsityCSC) = a.matrix

struct CartesianSparsity{A<:SparsityPattern,B,C} <: SparsityPattern
  sparsity::A
  bg_rows_to_act_rows::B
  bg_cols_to_act_cols::C
end

CartesianSparsity(a::SparsityPattern,::IdentityVector,::IdentityVector) = a

get_background_matrix(a::CartesianSparsity) = get_background_matrix(a.sparsity)

num_bg_rows(a::CartesianSparsity) = length(a.bg_rows_to_act_rows)
num_bg_cols(a::CartesianSparsity) = length(a.bg_cols_to_act_cols)

function bg_findnz(a::CartesianSparsity)
  I,J,V = findnz(a.sparsity)
  bg_rows = findall(!iszero,a.bg_rows_to_act_rows)
  bg_cols = findall(!iszero,a.bg_cols_to_act_cols)
  for k in eachindex(I)
    I[k] = bg_rows[I[k]]
    J[k] = bg_cols[J[k]]
  end
  return I,J,V
end

"""
    struct TProductSparsity{A<:SparsityPattern,B<:SparsityPattern} <: SparsityPattern
      sparsity::A
      sparsities_1d::Vector{B}
    end

Used to represent a sparsity pattern of a matrix obtained by integrating a
bilinear form on a triangulation that can be obtained as the tensor product of a
1-d triangulations. For example, this can be done when the mesh is Cartesian, and
the discretizing elements are cubes
"""
struct TProductSparsity{A<:SparsityPattern,B<:SparsityPattern} <: SparsityPattern
  sparsity::A
  sparsities_1d::Vector{B}
end

get_background_matrix(a::TProductSparsity) = get_background_matrix(a.sparsity)

univariate_num_rows(a::TProductSparsity) = Tuple(num_rows.(a.sparsities_1d))
univariate_num_cols(a::TProductSparsity) = Tuple(num_cols.(a.sparsities_1d))
univariate_findnz(a::TProductSparsity) = tuple_of_arrays(findnz.(a.sparsities_1d))
univariate_nnz(a::TProductSparsity) = Tuple(nnz.(a.sparsities_1d))

function get_sparse_dof_map(a::TProductSparsity,U::FESpace,V::FESpace,args...)
  Tu = get_dof_type(get_fe_dof_basis(U))
  Tv = get_dof_type(get_fe_dof_basis(V))
  full_ids = get_d_sparse_dofs_to_full_dofs(Tu,Tv,a)
  sparse_ids = to_nz_index(full_ids)
  SparseMatrixDofMap(sparse_ids,full_ids,a)
end

function get_d_sparse_dofs_to_full_dofs(Tu,Tv,a::TProductSparsity)
  I,J, = findnz(a)
  nrows = num_rows(a)
  ncols = num_cols(a)
  get_d_sparse_dofs_to_full_dofs(Tu,Tv,a,I,J,nrows,ncols)
end

function get_d_sparse_dofs_to_full_dofs(Tu,Tv,a::TProductSparsity{<:CartesianSparsity})
  I_bg,J_bg, = bg_findnz(a.sparsity)
  nrows_bg = num_bg_rows(a.sparsity)
  ncols_bg = num_bg_cols(a.sparsity)
  nrows = num_rows(a)
  dsd2sd = get_d_sparse_dofs_to_full_dofs(Tu,Tv,a,I_bg,J_bg,nrows_bg,ncols_bg)
  for (k,sdk) in enumerate(dsd2sd)
    if sdk > 0
      Ik_bg = fast_index(sdk,nrows_bg)
      Jk_bg = slow_index(sdk,nrows_bg)
      Ik = a.sparsity.bg_rows_to_act_rows[Ik_bg]
      Jk = a.sparsity.bg_cols_to_act_cols[Jk_bg]
      dsd2sd[k] = Ik+(Jk-1)*nrows
    end
  end
  return dsd2sd
end

function get_d_sparse_dofs_to_full_dofs(
  ::Type{<:Real},::Type{<:Real},a::TProductSparsity,I,J,nrows,ncols)
  _scalar_d_sparse_dofs_to_full_dofs(a,I,J,nrows,ncols)
end

function get_d_sparse_dofs_to_full_dofs(
  ::Type{Tu},::Type{Tv},a::TProductSparsity,I,J,nrows,ncols) where {Tu,Tv}
  _multivalue_d_sparse_dofs_to_full_dofs(a,I,J,nrows,ncols,num_components(Tu),num_components(Tv))
end

function _scalar_d_sparse_dofs_to_full_dofs(a::TProductSparsity,I,J,nrows,ncols)
  nnz_sizes = univariate_nnz(a)
  rows = univariate_num_rows(a)
  cols = univariate_num_cols(a)

  i,j, = univariate_findnz(a)
  d_to_nz_pairs = map((id,jd)->map(CartesianIndex,id,jd),i,j)

  D = length(a.sparsities_1d)
  cache = zeros(Int,D)
  dsd2sd = zeros(Int,nnz_sizes...)

  for k in eachindex(I)
    rows_1d = _index_to_d_indices(I[k],rows)
    cols_1d = _index_to_d_indices(J[k],cols)
    _row_col_pair_to_nz_index!(cache,rows_1d,cols_1d,d_to_nz_pairs)
    dsd2sd[cache...] = I[k]+(J[k]-1)*nrows
  end

  return dsd2sd
end

function _multivalue_d_sparse_dofs_to_full_dofs(a::TProductSparsity,I,J,nrows,ncols,ncomps_row,ncomps_col)
  nnz_sizes = univariate_nnz(a)
  rows_no_comps = univariate_num_rows(a)
  cols_no_comps = univariate_num_cols(a)
  nrows_no_comps = prod(rows_no_comps)
  ncols_no_comps = prod(cols_no_comps)
  ncomps = ncomps_row*ncomps_col

  i,j, = univariate_findnz(a)
  d_to_nz_pairs = map((id,jd)->map(CartesianIndex,id,jd),i,j)

  D = length(a.sparsities_1d)
  cache = zeros(Int,D)
  dsd2sd = zeros(Int,nnz_sizes...,ncomps)

  for k in eachindex(I)
    I_node,I_comp = _fast_and_slow_index(I[k],nrows_no_comps)
    J_node,J_comp = _fast_and_slow_index(J[k],ncols_no_comps)
    comp = I_comp+(J_comp-1)*ncomps_row
    rows_1d = _index_to_d_indices(I_node,rows_no_comps)
    cols_1d = _index_to_d_indices(J_node,cols_no_comps)
    _row_col_pair_to_nz_index!(cache,rows_1d,cols_1d,d_to_nz_pairs)
    dsd2sd[cache...,comp] = I[k]+(J[k]-1)*nrows
  end

  return dsd2sd
end

# utils

get_dof_type(b) = @abstractmethod
get_dof_type(b::LagrangianDofBasis{P,V}) where {P,V} = change_eltype(V,Float64)
get_dof_type(dof::CellDof) = get_dof_type(testitem(get_data(dof)))

function trivial_symbolic_loop_matrix!(A,cellidsrows,cellidscols)
  mat1 = nothing
  rows_cache = array_cache(cellidsrows)
  cols_cache = array_cache(cellidscols)

  rows1 = getindex!(rows_cache,cellidsrows,1)
  cols1 = getindex!(cols_cache,cellidscols,1)

  touch! = FESpaces.TouchEntriesMap()
  touch_cache = return_cache(touch!,A,mat1,rows1,cols1)
  caches = touch_cache,rows_cache,cols_cache

  for cell in 1:length(cellidscols)
    rows = getindex!(rows_cache,cellidsrows,cell)
    cols = getindex!(cols_cache,cellidscols,cell)
    evaluate!(touch_cache,touch!,A,mat1,rows,cols)
  end

  return A
end

function _get_common_domain(U::FESpace,V::FESpace)
  msg = """\n
  Cannot define a sparsity pattern object between the FE spaces given as input,
  as they are defined on incompatible triangulations
  """

  trian_U = get_triangulation(U)
  trian_V = get_triangulation(V)
  sa_tV = is_change_possible(trian_U,trian_V)
  sb_tU = is_change_possible(trian_V,trian_U)
  if sa_tV && sb_tU
    target_trian = best_target(trian_U,trian_V)
  elseif !sa_tV && sb_tU
    target_trian = trian_U
  elseif sa_tV && !sb_tU
    target_trian = trian_V
  else
    @notimplemented msg
  end
end

function _row_col_pair_to_nz_index!(
  cache,
  rows_1d::NTuple{D,Int},
  cols_1d::NTuple{D,Int},
  d_to_nz_pairs::Vector{Vector{CartesianIndex{2}}}
  ) where D

  for d in 1:D
    index = 0
    row_d = rows_1d[d]
    col_d = cols_1d[d]
    index_d = CartesianIndex((row_d,col_d))
    nz_pairs = d_to_nz_pairs[d]
    for (i,nzi) in enumerate(nz_pairs)
      if nzi == index_d
        index = i
        break
      end
    end
    @assert !iszero(index) "Could not build sparse dof mapping"
    cache[d] = index
  end
  return cache
end

function _fast_and_slow_index(i::Integer,s::Integer)
  fast_index(i,s),slow_index(i,s)
end

function _index_to_d_indices(i::Integer,s2::NTuple{2,Integer})
  _fast_and_slow_index(i,s2[1])
end

function _index_to_d_indices(i::Integer,sD::NTuple{D,Integer}) where D
  sD_minus_1 = sD[1:end-1]
  nD_minus_1 = prod(sD_minus_1)
  iD = slow_index(i,nD_minus_1)
  (_index_to_d_indices(i,sD_minus_1)...,iD)
end
