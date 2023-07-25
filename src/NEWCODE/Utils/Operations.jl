function expand(tup::Tuple)
  t = ()
  for el = tup
    if isa(el,Tuple)
      t = (t...,expand(el)...)
    else
      t = (t...,el)
    end
  end
  t
end

function SparseArrays.findnz(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
  numnz = nnz(S)
  I = Vector{Ti}(undef,numnz)
  J = Vector{Ti}(undef,numnz)
  V = Vector{Tv}(undef,numnz)

  count = 1
  @inbounds for col = 1:size(S,2), k = SparseArrays.getcolptr(S)[col] : (SparseArrays.getcolptr(S)[col+1]-1)
      I[count] = rowvals(S)[k]
      J[count] = col
      V[count] = nonzeros(S)[k]
      count += 1
  end

  nz = findall(x -> x .>= eps(),V)

  (I[nz],J[nz],V[nz])
end

function compress_array(entire_array::AbstractVector)
  nonzero_idx = findall(x -> abs(x) ≥ eps(),entire_array)
  nonzero_idx,entire_array[nonzero_idx]
end

function compress_array(entire_array::AbstractMatrix)
  sum_cols = reshape(sum(entire_array,dims=2),:)
  nonzero_idx = findall(x -> abs(x) ≥ eps(),sum_cols)
  nonzero_idx,entire_array[nonzero_idx,:]
end

function compress_array(entire_array::SparseMatrixCSC{Float,Int})
  findnz(entire_array[:])
end

function allocate_matrix(::EMatrix{T},sizes...) where T
  Elemental.zeros(EMatrix{T},sizes...)
end

function allocate_matrix(::Matrix{T},sizes...) where T
  zeros(T,sizes...)
end

function Base.getindex(
  emat::EMatrix{T},
  k1::Int,
  idx::Union{UnitRange{Int},Vector{Int},Colon}) where T

  reshape(convert(Matrix{T},emat[k1:k1,idx]),:)
end

function Base.getindex(
  emat::EMatrix{T},
  idx::Union{UnitRange{Int},Vector{Int},Colon},
  k2::Int) where T

  reshape(convert(Matrix{T},emat[idx,k2:k2]),:)
end

function Gridap.FESpaces.collect_cell_vector(
  test::FESpace,
  a::DomainContribution,
  trian::Triangulation)

  w = []
  r = []
  for strian in get_domains(a)
    if strian == trian
      scell_vec = get_contribution(a,strian)
      cell_vec,trian = move_contributions(scell_vec,strian)
      @assert ndims(eltype(cell_vec)) == 1
      cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
      rows = get_cell_dof_ids(test,trian)
      push!(w,cell_vec_r)
      push!(r,rows)
    end
  end
  (w,r)
end

function Gridap.FESpaces.collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  trian::Triangulation)

  w = []
  r = []
  c = []
  for strian in get_domains(a)
    if strian == trian
      scell_mat = get_contribution(a,strian)
      cell_mat,trian = move_contributions(scell_mat,strian)
      @assert ndims(eltype(cell_mat)) == 2
      cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
      cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
      rows = get_cell_dof_ids(test,trian)
      cols = get_cell_dof_ids(trial,trian)
      push!(w,cell_mat_rc)
      push!(r,rows)
      push!(c,cols)
    end
  end
  (w,r,c)
end

function collect_cell_contribution(
  test::FESpace,
  a::DomainContribution,
  trian::Triangulation)

  w = []
  for strian in get_domains(a)
    if strian == trian
      scell = get_contribution(a,strian)
      cell,trian = move_contributions(scell,strian)
      @assert ndims(eltype(cell)) == 1
      cell_r = attach_constraints_rows(test,cell,trian)
      push!(w,cell_r)
    end
  end
  first(w)
end

function collect_cell_contribution(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  trian::Triangulation)

  w = []
  for strian in get_domains(a)
    if strian == trian
      scell = get_contribution(a,strian)
      cell,trian = move_contributions(scell,strian)
      @assert ndims(eltype(cell)) == 2
      cell_c = attach_constraints_cols(trial,cell,trian)
      cell_rc = attach_constraints_rows(test,cell_c,trian)
      push!(w,cell_rc)
    end
  end
  first(w)
end

function collect_trian(a::DomainContribution)
  t = ()
  for trian in get_domains(a)
    t = (t...,trian)
  end
  unique(t)
end

function expand(tup::Tuple)
  t = ()
  for el = tup
    if isa(el,Tuple)
      t = (t...,expand(el)...)
    else
      t = (t...,el)
    end
  end
  t
end

function Base.:(==)(
  a::T,
  b::T
  ) where {T<:Union{Triangulation,Grid}}

  for field in propertynames(a)
    a_field = getproperty(a,field)
    b_field = getproperty(b,field)
    if isa(a_field,GridapType)
      (==)(a_field,b_field)
    else
      if isdefined(a_field,1) && !(==)(a_field,b_field)
        return false
      end
    end
  end
  return true
end

function is_parent(tparent::Triangulation,tchild::Triangulation)
  try
    try
      tparent.model == tchild.model && tparent.grid == tchild.grid.parent
    catch
      tparent == tchild.parent
    end
  catch
    false
  end
end

function modify_measures!(measures::Vector{Measure},m::Measure)
  for (nmeas,meas) in enumerate(measures)
    if is_parent(get_triangulation(meas),get_triangulation(m))
      measures[nmeas] = m
      return
    end
  end
  @unreachable "Unrecognizable measure"
end

function modify_measures(measures::Vector{Measure},m::Measure)
  new_measures = copy(measures)
  modify_measures!(new_measures,m)
  new_measures
end

Gridap.CellData.get_triangulation(m::Measure) = m.quad.trian

function Gridap.FESpaces.get_order(test::SingleFieldFESpace)
  basis = get_fe_basis(test)
  first(Gridap.FESpaces.get_order(first(basis.cell_basis.values).fields))
end

function Gridap.FESpaces.get_order(test::MultiFieldFESpace)
  orders = map(get_order,test)
  maximum(orders)
end

# Remove when possible
function Gridap.Geometry.is_change_possible(
  strian::Triangulation,
  ttrian::Triangulation)

  msg = """\n
  Triangulations do not point to the same background discrete model!
  """

  if strian === ttrian
    return true
  end

  @check get_background_model(strian) == get_background_model(ttrian) msg

  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  is_change_possible(sglue,tglue)
end
