function collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection)

  cell_irows = get_cellids_rows(hr)
  cell_icols = get_cellids_cols(hr)
  icells = get_owned_icells(hr)
  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_irows,cell_icols,icells)
end

function collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection)

  cell_irows = get_cellids_rows(hr)
  icells = get_owned_icells(hr)
  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_irows,icells)
end

# function separate_celldata(
#   hr::BlockHyperReduction,
#   celldata::AbstractArray,
#   cells::AbstractVector)

#   cells = get_integration_cells(hr)
#   block_c = get_owned_icells(hr,cells)
#   block_d = testitem(celldata)
#   @assert block_c.touched == block_d.touched
#   ki = SplitCellData(testitem(block_c))
#   k = Array{typeof(ki),ndims(block_c)}(undef,size(block_c))
#   for i in eachindex(block_c)
#     if block_c.touched[i]
#       k[i] = SplitCellData(block_c.array[i])
#     end
#   end
#   lazy_map(ArrayBlock(k,block_c.touched),celldata)
# end

struct BlockReindex{A} <: Map
  values::A
  blockid::Int
end

function Arrays.return_cache(k::BlockReindex,i...)
  array_cache(k.values)
end

function Arrays.evaluate!(cache,k::BlockReindex,i...)
  a = getindex!(cache,k.values,i...)
  a.array[k.blockid]
end

# function Arrays.lazy_map(k::BlockReindex{<:LazyArray},::Type{T},j_to_i::AbstractArray) where T
#   i_to_maps = k.values.maps
#   i_to_args = k.values.args
#   j_to_maps = lazy_map(Reindex(i_to_maps),eltype(i_to_maps),j_to_i)
#   j_to_args = map(i_to_fk->lazy_map(BlockReindex(i_to_fk,k.blockid),eltype(i_to_fk),j_to_i),i_to_args)
#   LazyArray(T,j_to_maps,j_to_args...)
# end

# function Arrays.return_cache(k::SplitCellData,data::AbstractArray)
#   return_cache(Reindex(data),k.ids)
# end

# function Arrays.evaluate!(cache,k::SplitCellData,data::AbstractArray)
#   evaluate!(cache,Reindex(data),k.ids)
# end

# function Arrays.return_cache(k::SplitCellData,f::ParamBlock)
#   di = testitem(f)
#   ci = return_cache(k,di)
#   vi = evaluate!(ci,k,di)
#   cache = Vector{typeof(ci)}(undef,param_length(f))
#   array = Vector{typeof(vi)}(undef,param_length(f))
#   for i in param_eachindex(f)
#     cache[i] = return_cache(k,param_getindex(f,i))
#   end
#   return GenericParamBlock(array),cache
# end

# function Arrays.evaluate!(cache,k::SplitCellData,f::ParamBlock)
#   array,c = cache
#   for i in param_eachindex(f)
#     array.data[i] = evaluate!(c[i],k,param_getindex(f,i))
#   end
#   array
# end

# function Arrays.return_cache(k::SplitBlockCellData,data::ArrayBlock{A,N}) where {A,N}
#   di = data.array[k.blockid]
#   ki = Reindex(di)
#   ci = return_cache(ki,k.ids)
#   vi = evaluate!(ci,ki,k.ids)
#   cache = Array{typeof(ci),N}(undef,size(data))
#   array = Array{typeof(vi),N}(undef,size(data))
#   for i in eachindex(data.array)
#     if data.touched[i]
#       cache[i] = return_cache(k.array[i],data.array[i])
#     end
#   end
#   return ArrayBlock(array,data.touched),cache
# end

# function Arrays.evaluate!(cache,k::ArrayBlock{<:SplitCellData,N},data::ArrayBlock{A,N}) where {A,N}
#   @check k.touched == data.touched
#   a,c = cache
#   for i in eachindex(data.array)
#     if data.touched[i]
#       a.array[i] = evaluate!(c[i],k.array[i],data.array[i])
#     end
#   end
#   a
# end

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

function assemble_hr_vector_add!(b::ArrayBlock,cellvec,cellidsrows::ArrayBlock,icells::ArrayBlock)
  @check b.touched == cellidsrows.touched == icells.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellveci = lazy_map(BlockReindex(cellvec,i),icells.array[i])
      assemble_hr_vector_add!(b.array[i],cellveci,cellidsrows.array[i])
    end
  end
end

function assemble_hr_vector_add!(b,cellvec,cellidsrows,args...)
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
  A::ArrayBlock,cellmat,cellidsrows::ArrayBlock,cellidscols::ArrayBlock,icells::ArrayBlock)
  @check A.touched == cellidsrows.touched == cellidscols.touched == icells.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellmati = lazy_map(BlockReindex(cellmat,i),icells.array[i])
      assemble_hr_matrix_add!(A.array[i],cellmati,cellidsrows.array[i],cellidscols.array[i])
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
