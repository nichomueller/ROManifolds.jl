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

function assemble_hr_vector_add!(b,a::SparseMatrixAssembler,vecdata)
  assemble_vector_add!(b,a,vecdata)
end

function assemble_hr_matrix_add!(A,a::SparseMatrixAssembler,matdata)
  strategy = FESpaces.get_assembly_strategy(a)
  for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = FESpaces.map_cell_rows(strategy,_cellidsrows)
    cellidscols = FESpaces.map_cell_cols(strategy,_cellidscols)
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
