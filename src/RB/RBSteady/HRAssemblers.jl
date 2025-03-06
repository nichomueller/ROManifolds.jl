function collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection)

  cells = Utils.get_tface_to_mface(strian)
  cell_irows = get_cellids_rows(hr)
  cell_icols = get_cellids_cols(hr)
  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  cell_hr_mat_rc = separate_celldata(hr,cell_mat_rc,cells)
  (cell_hr_mat_rc,cell_irows,cell_icols)
end

function collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection)

  cells = Utils.get_tface_to_mface(strian)
  cell_irows = get_cellids_rows(rhs_trian)
  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  cell_hr_vec_r = separate_celldata(hr,cell_vec_r,cells)
  (cell_hr_vec_r,cell_irows)
end

function separate_celldata(
  hr::HyperReduction,
  celldata::AbstractArray,
  cells::AbstractVector)

  celldata
end

function separate_celldata(
  hr::BlockHyperReduction,
  celldata::AbstractArray,
  cells::AbstractVector)

  block_c = get_owned_icells(hr,cells)
  block_d = testitem(celldata)
  @assert block_c.touched == block_d.touched
  ki = SplitCellData(testitem(block_c))
  k = Array{typeof(ki),ndims(block_c)}(undef,size(block_c))
  for i in eachindex(block_c)
    if block_c.touched[i]
      k[i] = SplitCellData(block_c.array[i])
    end
  end
  lazy_map(ArrayBlock(k,block_c.touched),celldata)
end

struct SplitCellData{T<:Integer} <: Map
  ids::Vector{T}
end

function Arrays.return_cache(k::SplitCellData,data::AbstractArray)
  return_cache(Reindex(data),k.ids)
end

function Arrays.evaluate!(cache,k::SplitCellData,data::AbstractArray)
  evaluate!(cache,Reindex(data),k.ids)
end

function Arrays.return_cache(k::ArrayBlock{<:SplitCellData,N},data::ArrayBlock{A,N}) where {A,N}
  @check k.touched == data.touched
  ki = testitem(k)
  di = testitem(data)
  ci = return_cache(ki,di)
  vi = evaluate!(ci,ki,di)
  cache = Array{typeof(ci),N}(undef,size(data))
  array = Array{typeof(vi),N}(undef,size(data))
  for i in eachindex(data.array)
    if data.touched[i]
      cache[i] = return_cache(k.array[i],data.array[i])
    end
  end
  return ArrayBlock(array,data.touched),cache
end

function Arrays.evaluate!(cache,k::ArrayBlock{<:SplitCellData,N},data::ArrayBlock{A,N}) where {A,N}
  @check k.touched == data.touched
  a,c = cache
  for i in eachindex(data.array)
    if data.touched[i]
      a.array[i] = evaluate!(c[i],k.array[i],data.array[i])
    end
  end
  a
end

struct AddHREntriesMap{F}
  combine::F
end

function Arrays.return_cache(k::AddHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),param_length(vs))
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,vs,is)
  add_hr_entries!(cache,k.combine,A,vs,is)
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,vs,is,js)
  add_hr_entries!(cache,k.combine,A,vs,is,js)
end

function Arrays.return_cache(
  k::AddHREntriesMap,A,v::MatrixBlock,I::VectorBlock,J::VectorBlock)

  qs = findall(v.touched)
  i,j = Tuple(first(qs))
  cij = return_cache(k,A,v.array[i,j],I.array[i],J.array[j])
  ni,nj = size(v.touched)
  cache = Matrix{typeof(cij)}(undef,ni,nj)
  for j in 1:nj
    for i in 1:ni
      if v.touched[i,j]
        cache[i,j] = return_cache(k,A,v.array[i,j],I.array[i],J.array[j])
      end
    end
  end
  cache
end

function Arrays.evaluate!(
  cache,k::AddHREntriesMap,A,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
  ni,nj = size(v.touched)
  for j in 1:nj
    for i in 1:ni
      if v.touched[i,j]
        evaluate!(cache[i,j],k,A,v.array[i,j],I.array[i],J.array[j])
      end
    end
  end
end

function Arrays.return_cache(
  k::AddHREntriesMap,A,v::VectorBlock,I::VectorBlock)

  qs = findall(v.touched)
  i = first(qs)
  ci = return_cache(k,A,v.array[i],I.array[i])
  ni = length(v.touched)
  cache = Vector{typeof(ci)}(undef,ni)
  for i in 1:ni
    if v.touched[i]
      cache[i] = return_cache(k,A,v.array[i],I.array[i])
    end
  end
  cache
end

function Arrays.evaluate!(
  cache,k::AddHREntriesMap,A,v::VectorBlock,I::VectorBlock)
  ni = length(v.touched)
  for i in 1:ni
    if v.touched[i]
      evaluate!(cache[i],k,A,v.array[i],I.array[i])
    end
  end
end

@inline function add_hr_entries!(
  vij,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,js)

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          if i == j
            get_param_entry!(vij,vs,li,lj)
            add_entry!(combine,A,vij,i)
          end
        end
      end
    end
  end
  A
end

@inline function add_hr_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is)

  for (li,i) in enumerate(is)
    if i>0
      get_param_entry!(vi,vs,li)
      add_entry!(combine,A,vi,i)
    end
  end
  A
end

function assemble_hr_vector_add!(b::BlockParamArray,cellvec::ArrayBlock,cellidsrows::ArrayBlock)
  for i in eachindex(cellvec)
    if cellvec.touched[i]
      assemble_hr_vector_add!(blocks(b)[i],cellvec.array[i],cellidsrows.array[i])
    end
  end
end

function assemble_hr_vector_add!(b,cellvec,cellidsrows)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = AddHREntriesMap(+)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add_cache,vals_cache,rows_cache
    _numeric_loop_hr_vector!(b,caches,cellvec,cellidsrows)
  end
  b
end

@noinline function _numeric_loop_hr_vector!(vec,caches,cell_vals,cell_rows)
  FESpaces._numeric_loop_vector!(vec,caches,cell_vals,cell_rows)
end

function assemble_hr_matrix_add!(
  A::BlockParamArray,cellmat::ArrayBlock,cellidsrows::ArrayBlock,cellidscols::ArrayBlock)
  for i in eachindex(cellmat)
    if cellmat.touched[i]
      assemble_hr_matrix_add!(
        blocks(A)[i],cellmat.array[i],cellidsrows.array[i],cellidscols.array[i])
    end
  end
end

function assemble_hr_matrix_add!(A,cellmat,cellidsrows,cellidscols)
  @assert length(cellidscols) == length(cellidsrows)
  @assert length(cellmat) == length(cellidsrows)
  if length(cellmat) > 0
    rows_cache = array_cache(cellidsrows)
    cols_cache = array_cache(cellidscols)
    vals_cache = array_cache(cellmat)
    mat1 = getindex!(vals_cache,cellmat,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    cols1 = getindex!(cols_cache,cellidscols,1)
    add! = AddHREntriesMap(+)
    add_cache = return_cache(add!,A,mat1,rows1,cols1)
    caches = add_cache,vals_cache,rows_cache,cols_cache
    _numeric_loop_hr_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
  end
  A
end

@noinline function _numeric_loop_hr_matrix!(mat,caches,cell_vals,cell_rows,cell_cols)
  add_cache,vals_cache,rows_cache,cols_cache = caches
  add! = AddHREntriesMap(+)
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,mat,vals,rows,cols)
  end
end

# # utils

# struct HRAssemblyStrategy <: AssemblyStrategy end

# FESpaces.map_cell_rows(k::HRAssemblyStrategy,cell_ids) = cell_ids
# FESpaces.map_cell_cols(k::HRAssemblyStrategy,cell_ids) = cell_ids
# FESpaces.map_cell_rows(k::HRAssemblyStrategy,cell_ids::ArrayBlock) = LazyArrayBlock(cell_ids)
# FESpaces.map_cell_cols(k::HRAssemblyStrategy,cell_ids::ArrayBlock) = LazyArrayBlock(cell_ids)

# struct LazyArrayBlock{A,N,T} <: AbstractVector{ArrayBlock{T,N}}
#   blocks::ArrayBlock{A,N}
#   l::Int
#   function LazyArrayBlock(blocks::ArrayBlock{A,N},l::Int) where {A,N}
#     T = eltype(A)
#     new{A,N,T}(blocks,l)
#   end
# end

# function LazyArrayBlock(blocks::ArrayBlock)
#   l = _max_length(blocks)
#   LazyArrayBlock(blocks,l)
# end

# Base.size(b::LazyArrayBlock) = (b.l,)

# function Base.getindex(b::LazyArrayBlock,j::Int)
#   cache = array_cache(b)
#   getindex!(cache,b,j)
# end

# function Arrays.array_cache(a::LazyArrayBlock{T,N}) where {T,N}
#   blocks = a.blocks
#   ai = testitem(blocks)
#   ci = array_cache(ai)
#   vi = getindex!(ci,ai,1)
#   c = Array{typeof(ci),N}(undef,size(blocks))
#   v = Array{typeof(vi),N}(undef,size(blocks))
#   for i in eachindex(blocks)
#     if blocks.touched[i]
#       c[i] = array_cache(blocks.array[i])
#     end
#   end
#   ArrayBlock(v,blocks.touched),c
# end

# function Arrays.getindex!(cache,a::LazyArrayBlock,j::Int)
#   v,c = cache
#   blocks = a.blocks
#   for i in eachindex(blocks)
#     if blocks.touched[i]
#       bi = blocks.array[i]
#       li = length(bi)
#       if j <= li
#         v[i] = getindex!(c[i],bi,j)
#       else
#         v0 = getindex!(c[i],bi,li)
#         fill!(v0,zero(eltype(v0)))
#         v[i] = v0
#       end
#     end
#   end
#   v
# end

# function _max_length(blocks::ArrayBlock)
#   l = 0
#   for i in eachindex(blocks)
#     if blocks.touched[i]
#       l = max(l,length(blocks.array[i]))
#     end
#   end
#   l
# end
