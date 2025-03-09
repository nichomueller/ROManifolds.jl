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
  ::KroneckerTransientHR,
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection,
  common_indices::AbstractVector)

  cell_irows = get_cellids_rows(hr)
  cell_icols = get_cellids_cols(hr)
  icells = get_owned_icells(hr)
  _,locations = get_indices_locations(hr,common_indices)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_irows,cell_icols,icells,locations)
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
  (cell_mat_rc,cell_irows,cell_icols,icells,indices,locations)
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
  ::KroneckerTransientHR,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  hr::Projection,
  common_indices::AbstractVector)

  cell_irows = get_cellids_rows(hr)
  icells = get_owned_icells(hr)
  _,locations = get_indices_locations(hr,common_indices)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_irows,icells,locations)
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

@inline function add_kron_entry!(
  combine::Function,A::ConsecutiveParamVector,v::Number,hr_indices,i)

  n_hr_ids = length(hr_indices)
  data = get_all_data(A)
  for (k,hr_id_k) in enumerate(hr_indices)
    i_hr = (k-1)*n_hr_ids + i
    aik = data[i_hr,k]
    data[i_hr,k] = combine(aik,v)
  end
  A
end

@inline function add_kron_entry!(
  combine::Function,A::ConsecutiveParamVector,v::AbstractVector,hr_indices::Range1D,i)

  param_ids = hr_indices.parent.axis1
  hr_time_ids = hr_indices.parent.axis2
  n_hr_time_ids = length(hr_time_ids)
  data = get_all_data(A)
  for (k,hr_id_k) in enumerate(hr_indices)
    i_hr = (hr_id_k-1)*n_hr_time_ids + i
    aik = data[i_hr]
    vk = v[k]
    data[i_hr] = combine(aik,vk)
  end
  A
end

abstract type AddPairedHREntriesMap <: Map end

struct AddKroneckerHREntriesMap{F,L<:AbstractVector} <: AddPairedHREntriesMap
  combine::F
  locations::L
end

function Arrays.return_cache(k::AddKroneckerHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),length(k.locations))
end

function Arrays.evaluate!(cache,k::AddKroneckerHREntriesMap,A,vs,is)
  add_kronecker_hr_entries!(cache,k.combine,A,vs,is,k.locations)
end

function Arrays.evaluate!(cache,k::AddKroneckerHREntriesMap,A,vs,is,js)
  add_kronecker_hr_entries!(cache,k.combine,A,vs,is,js,k.locations)
end

@inline function add_kronecker_hr_entries!(
  vij,combine::Function,A::AbstractParamVector,vs,is,js,lt)

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          vij = vs[li,lj]
          add_kron_entry!(combine,A,vij,lt,i)
        end
      end
    end
  end
  A
end

@inline function add_kronecker_hr_entries!(
  vi,combine::Function,A::AbstractParamVector,vs,is,lt)

  for (li,i) in enumerate(is)
    if i>0
      vi = vs[li]
      add_kron_entry!(combine,A,vi,lt,i)
    end
  end
  A
end

@inline function add_kronecker_hr_entries!(
  vij,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,js,lt)

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          get_hr_param_entry!(vij,vs,li,lj)
          add_kron_entry!(combine,A,vij,lt,i)
        end
      end
    end
  end
  A
end

@inline function add_kronecker_hr_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,lt)

  for (li,i) in enumerate(is)
    if i>0
      get_param_entry!(vi,vs,li)
      add_kron_entry!(combine,A,vi,lt,i)
    end
  end
  A
end

struct AddLinearHREntriesMap{F,I<:AbstractVector,L<:AbstractVector} <: AddPairedHREntriesMap
  combine::F
  indices::I
  locations::L
end

function Arrays.return_cache(k::AddLinearHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),length(k.locations))
end

function Arrays.evaluate!(cache,k::AddLinearHREntriesMap,A,vs,is)
  add_linear_hr_entries!(cache,k.combine,A,vs,is,k.locations,k.indices)
end

function Arrays.evaluate!(cache,k::AddLinearHREntriesMap,A,vs,is,js)
  add_linear_hr_entries!(cache,k.combine,A,vs,is,js,k.locations,k.indices)
end

@inline function add_linear_hr_entries!(
  vij,combine::Function,A::AbstractParamVector,vs,is,js,lt,t)

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

@inline function add_linear_hr_entries!(
  vi,combine::Function,A::AbstractParamVector,vs,is,lt,t)

  for (li,i) in enumerate(is)
    if i>0
      if i == t
        vi = vs[li]
        add_entry!(combine,A,vi,i)
      end
    end
  end
  A
end

@inline function add_linear_hr_entries!(
  vij,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,js,lt,t)

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          if i == j == t
            vij = vs[li,lj]
            add_entry!(combine,A,vij,i)
          end
        end
      end
    end
  end
  A
end

@inline function add_linear_hr_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,lt,t)

  for (li,i) in enumerate(is)
    if i>0
      if i == t
        get_hr_param_entry!(vi,vs,lt,li)
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
  locations::ArrayBlock)

  @check cellidsrows.touched == icells.touched == locations.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellveci = lazy_map(BlockReindex(cellvec,i),icells.array[i])
      assemble_hr_vector_add!(
        b.array[i],cellveci,cellidsrows.array[i],icells.array[i],locations.array[i])
    end
  end
end

function RBSteady.assemble_hr_vector_add!(b,cellvec,cellidsrows,icells,locations)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = AddKroneckerHREntriesMap(+,locations)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add!,add_cache,vals_cache,rows_cache
    RBSteady._numeric_loop_hr_vector!(b,caches,cellvec,cellidsrows)
  end
  b
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
        b.array[i],cellveci,cellidsrows.array[i],icells.array[i],indices.array[i],locations.array[i])
    end
  end
end

function RBSteady.assemble_hr_vector_add!(b,cellvec,cellidsrows,icells,indices,locations)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = AddLinearHREntriesMap(+,indices,locations)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add!,add_cache,vals_cache,rows_cache
    RBSteady._numeric_loop_hr_vector!(b,caches,cellvec,cellidsrows)
  end
  b
end


function RBSteady.assemble_hr_matrix_add!(
  A::ArrayBlock,
  cellmat,
  cellidsrows::ArrayBlock,
  cellidscols::ArrayBlock,
  icells::ArrayBlock,
  locations::ArrayBlock)

  @check (cellidsrows.touched == cellidscols.touched == icells.touched == locations.touched)
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellmati = lazy_map(BlockReindex(cellmat,i),icells.array[i])
      assemble_hr_matrix_add!(
        A.array[i],cellmati,cellidsrows.array[i],cellidscols.array[i],icells.array[i],
        locations.array[i])
    end
  end
end

function RBSteady.assemble_hr_matrix_add!(A,cellmat,cellidsrows,cellidscols,icells,locations)
  @assert length(cellidscols) == length(cellidsrows)
  @assert length(cellmat) == length(cellidsrows)
  if length(cellmat) > 0
    rows_cache = array_cache(cellidsrows)
    cols_cache = array_cache(cellidscols)
    vals_cache = array_cache(cellmat)
    mat1 = getindex!(vals_cache,cellmat,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    cols1 = getindex!(cols_cache,cellidscols,1)
    add! = AddKroneckerHREntriesMap(+,locations)
    add_cache = return_cache(add!,A,mat1,rows1,cols1)
    caches = add!,add_cache,vals_cache,rows_cache,cols_cache
    RBSteady._numeric_loop_hr_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
  end
  A
end

function RBSteady.assemble_hr_matrix_add!(
  A::ArrayBlock,
  cellmat,
  cellidsrows::ArrayBlock,
  cellidscols::ArrayBlock,
  icells::ArrayBlock,
  indices::ArrayBlock,
  locations::ArrayBlock)

  @check (cellidsrows.touched == cellidscols.touched == icells.touched ==
    indices.touched == locations.touched)
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellmati = lazy_map(BlockReindex(cellmat,i),icells.array[i])
      assemble_hr_matrix_add!(
        A.array[i],cellmati,cellidsrows.array[i],cellidscols.array[i],icells.array[i],
        indices.array[i],locations.array[i])
    end
  end
end

function RBSteady.assemble_hr_matrix_add!(
  A,cellmat,cellidsrows,cellidscols,icells,indices,locations)

  @assert length(cellidscols) == length(cellidsrows)
  @assert length(cellmat) == length(cellidsrows)
  if length(cellmat) > 0
    rows_cache = array_cache(cellidsrows)
    cols_cache = array_cache(cellidscols)
    vals_cache = array_cache(cellmat)
    mat1 = getindex!(vals_cache,cellmat,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    cols1 = getindex!(cols_cache,cellidscols,1)
    add! = AddLinearHREntriesMap(+,indices,locations)
    add_cache = return_cache(add!,A,mat1,rows1,cols1)
    caches = add!,add_cache,vals_cache,rows_cache,cols_cache
    RBSteady._numeric_loop_hr_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
  end
  A
end
