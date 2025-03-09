abstract type TransientHRStyle end
struct KroneckerTransientHR <: TransientHRStyle end
struct LinearTransientHR <: TransientHRStyle end

TransientHRStyle(hr::HyperReduction) = LinearTransientHR()
TransientHRStyle(hr::TransientHyperReduction) = KroneckerTransientHR()
TransientHRStyle(hr::BlockProjection) = TransientHRStyle(testitem(hr))

function RBSteady.collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection,
  common_indices::AbstractVector)

  _collect_cell_hr_matrix(TransientHRStyle(hr),trial,test,a,strian,hr,common_indices)
end

function _collect_cell_hr_matrix(
  ::TransientHRStyle,
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection,
  common_indices::AbstractVector)

  collect_cell_hr_matrix(trial,test,a,strian,hr)
end

function _collect_cell_hr_matrix(
  ::LinearTransientHR,
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection,
  common_indices::AbstractVector)

  cell_irows = get_cellids_rows(hr)
  cell_icols = get_cellids_cols(hr)
  icells = get_owned_icells(hr)
  indices,locations = get_indices_locations(hr,common_indices)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_irows,cell_icols,icells,times,indices,locations)
end

function RBSteady.collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection,
  common_indices::AbstractVector)

  _collect_cell_hr_vector(TransientHRStyle(hr),test,a,strian,hr,common_indices)
end

function _collect_cell_hr_vector(
  ::TransientHRStyle,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection,
  common_indices::AbstractVector)

  collect_cell_hr_vector(test,a,strian,hr)
end

function _collect_cell_hr_vector(
  ::LinearTransientHR,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection,
  common_indices::AbstractVector)

  cell_irows = get_cellids_rows(hr)
  icells = get_owned_icells(hr)
  indices,locations = get_indices_locations(hr,common_indices)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_irows,icells,indices,locations)
end

function get_hr_param_entry!(v::AbstractVector,b::GenericParamBlock,hr_indices,i...)
  for (k,hrk) in eachindex(hr_indices)
    @inbounds v[k] = b.data[hrk][i...]
  end
  v
end

function get_hr_param_entry!(v::AbstractVector,b::TrivialParamBlock,hr_indices,i...)
  vk = b.data[i...]
  fill!(v,vk)
end

struct AddPairedHREntriesMap{F,I<:AbstractVector,L<:AbstractVector}
  combine::F
  indices::I
  locations::L
end

function Arrays.return_cache(k::AddPairedHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),length(k.indices))
end

function Arrays.evaluate!(cache,k::AddPairedHREntriesMap,A,vs,is)
  add_paired_hr_entries!(cache,k.combine,A,vs,is,k.locations,k.indices)
end

function Arrays.evaluate!(cache,k::AddPairedHREntriesMap,A,vs,is,js)
  add_paired_hr_entries!(cache,k.combine,A,vs,is,js,k.locations,k.indices)
end

@inline function add_paired_hr_entries!(
  vij,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,js,lt,t)

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          if i == j == t
            get_hr_param_entry!(vij,vs,lt,li,lj)
            add_entry!(combine,A,vij,i)
          end
        end
      end
    end
  end
  A
end

@inline function add_paired_hr_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,t)

  for (li,i) in enumerate(is)
    if i>0
      if i == t
        get_hr_param_entry!(vij,vs,lt,li)
        add_entry!(combine,A,vi,i)
      end
    end
  end
  A
end

function RBSteady.assemble_hr_vector_add!(
  b::ArrayBlock,
  cellvec,
  cellidsrows::ArrayBlock,
  icells::ArrayBlock,
  indices::ArrayBlock,
  locations::ArrayBlock)

  @check cellidsrows.touched == icells.touched == indices.touched == locations.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellveci = lazy_map(BlockReindex(cellvec,i),icells.array[i])
      assemble_hr_vector_add!(
        b.array[i],cellveci,cellidsrows.array[i],indices.array[i],locations.array[i])
    end
  end
end

function RBSteady.assemble_hr_vector_add!(b,cellvec,cellidsrows,indices,locations)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = AddPairedHREntriesMap(+,indices,locations)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add!,add_cache,vals_cache,rows_cache
    _numeric_loop_paired_hr_vector!(b,caches,cellvec,cellidsrows)
  end
  b
end

@noinline function _numeric_loop_paired_hr_vector!(vec,caches,cell_vals,cell_rows)
  add!,add_cache,vals_cache,rows_cache = caches
  @assert length(cell_vals) == length(cell_rows)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,vec,vals,rows)
  end
end

function RBSteady.assemble_hr_matrix_add!(
  A::ArrayBlock,
  cellmat,
  cellidsrows::ArrayBlock,
  cellidscols::ArrayBlock,
  icells::ArrayBlock,
  indices::ArrayBlock,
  locations::ArrayBlock)

  @check cellidsrows.touched == cellidscols.touched == icells.touched == indices.touched == locations.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellmati = lazy_map(BlockReindex(cellmat,i),icells.array[i])
      assemble_hr_matrix_add!(
        A.array[i],cellmati,cellidsrows.array[i],cellidscols.array[i],indices.array[i],locations.array[i])
    end
  end
end

function RBSteady.assemble_hr_matrix_add!(A,cellmat,cellidsrows,cellidscols,indices,locations)
  @assert length(cellidscols) == length(cellidsrows)
  @assert length(cellmat) == length(cellidsrows)
  if length(cellmat) > 0
    rows_cache = array_cache(cellidsrows)
    cols_cache = array_cache(cellidscols)
    vals_cache = array_cache(cellmat)
    mat1 = getindex!(vals_cache,cellmat,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    cols1 = getindex!(cols_cache,cellidscols,1)
    add! = AddPairedHREntriesMap(+,indices,locations)
    add_cache = return_cache(add!,A,mat1,rows1,cols1)
    caches = add!,add_cache,vals_cache,rows_cache,cols_cache
    _numeric_loop_paired_hr_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
  end
  A
end

@noinline function _numeric_loop_paired_hr_matrix!(mat,caches,cell_vals,cell_rows,cell_cols)
  add!,add_cache,vals_cache,rows_cache,cols_cache = caches
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,mat,vals,rows,cols)
  end
end
