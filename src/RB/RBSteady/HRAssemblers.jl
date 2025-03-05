# struct AddHREntriesMap <: Map
#   combine::Function
#   ids::Vector{Int}
#   i_to_ids::Vector{Int}
# end

# function AddHREntriesMap(combine::Function,ids::Vector{Int})
#   i_to_ids = sortperm(ids)
#   AddHREntriesMap(i_to_ids,ids,i_to_ids)
# end

# function AddHREntriesMap(ids::Vector{Int},args...)
#   AddHREntriesMap(+,ids)
# end

# function Arrays.return_cache(k::AddHREntriesMap,args...)
#   return_cache(Fields.AddEntriesMap(k.combine),args...)
# end

# function Arrays.evaluate!(cache,k::AddHREntriesMap,args...)
#   add_entries!(cache,k,args...)
# end

# @inline function Algebra.add_entries!(vi,k::AddHREntriesMap,A,vs::ParamBlock,is)
#   for (li,i) in enumerate(is)
#     if i>0
#       for (q,qi) in enumerate(k.ids)
#         if qi == i
#           get_param_entry!(vi,vs,q)
#           add_entry!(k.combine,A,vi,i)
#         end
#       end
#     end
#   end
#   A
# end

# @inline function Algebra.add_entries!(vi,k::AddHREntriesMap,A,vs::ParamBlock,is,js)
#   for (li,i) in enumerate(is)
#     if i>0
#       for (q,qi) in enumerate(k.ids)
#         if qi == i
#           get_param_entry!(vi,vs,q)
#           add_entry!(k.combine,A,vi,i)
#         end
#       end
#     end
#   end
#   A
# end
"""
    hr_jacobian!(
      A::HRParamArray,
      op::RBOperator,
      r::AbstractRealization,
      u,
      paramcache
      ) -> Nothing

Full order residual computed via integration on the [`IntegrationDomain`](@ref)
relative to the LHS defined in `op`
"""
function hr_jacobian!(
  A,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op.op)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  trian_jac = get_domains_jac(op.op)
  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)

  map(trian_jac) do strian
    A_trian = A.fecache[strian]
    i_trian = get_integration_domain(a[strian])
    scell_mat = get_contribution(dc,strian)
    cell_mat,trian = move_contributions(scell_mat,strian)
    @assert ndims(eltype(cell_mat)) == 2
    cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
    cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
    matdata = [cell_mat_rc],[i_trian.cell_irows],[i_trian.cell_icols]
    assemble_matrix_add!(b_trian,assem,matdata)
  end

  A
end

"""
    hr_residual!(
      b,
      op::RBOperator,
      r::AbstractRealization,
      u,
      paramcache
      ) -> Nothing

Full order residual computed via integration on the [`IntegrationDomain`](@ref)
relative to the RHS defined in `op`
"""
function hr_residual!(
  b::HRParamArray,
  op::ParamOperator,
  a::AffineContribution,
  r::Realization,
  u::AbstractParamVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  trian_res = get_domains_res(op.op)
  res = get_res(op.op)
  dc = res(μ,uh,v)

  map(trian_res) do strian
    b_trian = b.fecache[strian]
    i_trian = get_integration_domain(a[strian])
    scell_vec = get_contribution(dc,strian)
    cell_vec,trian = move_contributions(scell_vec,strian)
    @assert ndims(eltype(cell_vec)) == 1
    cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
    vecdata = [cell_vec_r],[i_trian.cell_irows]
    assemble_vector_add!(b_trian,assem,vecdata)
  end

  b
end

# function assemble_hr_vector_add!(b,a::SparseMatrixAssembler,vecdata,ids)
#   strategy = FESpaces.get_assembly_strategy(a)
#   _cellvec,__cellids = vecdata
#   cellvec,_cellids = first(_cellvec),first(__cellids)
#   cellids = FESpaces.map_cell_rows(strategy,_cellids)
#   if length(cellvec) > 0
#     rows_cache = array_cache(cellids)
#     vals_cache = array_cache(cellvec)
#     vals1 = getindex!(vals_cache,cellvec,1)
#     rows1 = getindex!(rows_cache,cellids,1)
#     add! = FESpaces.AddEntriesMap(+)
#     add_cache = return_cache(add!,b,vals1,rows1)
#     caches = add_cache,vals_cache,rows_cache
#     _numeric_loop_hr_vector!(b,caches,cellvec,cellids,ids)
#   end
#   b
# end

# @noinline function _numeric_loop_hr_vector!(vec,caches,cell_vals,cell_irows)
#   add_cache,vals_cache,irows_cache = caches
#   for cell in 1:length(cell_irows)
#     irows = getindex!(irows_cache,cell_irows,cell)
#     vals = getindex!(vals_cache,cell_vals,cell)
#     for (li,i) in enumerate(irows)
#       if i>0
#         vij = vs[li,lj]
#         add_entry!(combine,A,vij,i,j)
#       end
#     end
#     evaluate!(add_cache,add!,vec,vals,rows)
#   end
# end

# @noinline function _numeric_loop_hr_matrix!(
#   mat,caches,cell_vals,cell_rows,hr_rows,hr_cols,hr_cells)

#   # i_to_hr_rows = sortperm(hr_rows)
#   # i_to_hr_cols = sortperm(hr_cols)
#   add_cache,vals_cache,rows_cache,cols_cache = caches
#   @assert length(cell_vals) == length(cell_rows) == length(hr_cells)
#   @assert length(hr_rows) == length(hr_cols)
#   for k in eachindex(hr_rows)
#     hr_row = hr_rows[k]
#     hr_col = hr_cols[k]
#     hr_cell
#   end
#   for (hr_row,hr_col) in zip(hr_rows,hr_cols)
#     hr_cells[k]
#   end
#   for cell in 1:length(cell_rows)
#     rows = getindex!(rows_cache,cell_rows,cell)
#     vals = getindex!(vals_cache,cell_vals,cell)
#     evaluate!(add_cache,add!,vec,vals,rows,ids,oids)
#   end
# end
