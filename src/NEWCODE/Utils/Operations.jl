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

function collect_trian(a::DomainContribution)
  t = ()
  for trian in get_domains(a)
    t = (t...,trian)
  end
  unique(t)
end

function Base.:(==)(g1::UnstructuredGrid,g2::UnstructuredGrid)
  (g1.node_coordinates == g2.node_coordinates &&
  g1.cell_node_ids == g2.cell_node_ids &&
  g1.reffes == g2.reffes &&
  g1.cell_types == g2.cell_types &&
  g1.orientation_style == g2.orientation_style &&
  g1.facet_normal == g2.facet_normal)
end

function Base.:(==)(t1::Triangulation,t2::Triangulation)
  t1.model == t2.model && t1.grid == t2.grid
end

function is_parent(
  tparent::Triangulation,
  tchild::BodyFittedTriangulation{Dt,Dp,A,<:GridView,C}) where {Dt,Dp,A,C}

  tparent.model == tchild.model && tparent.grid == tchild.grid.parent
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

# Remove when possible
function Gridap.Geometry.is_change_possible(
  strian::Triangulation,
  ttrian::Triangulation)

  if strian === ttrian
    return true
  end

  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  is_change_possible(sglue,tglue)
end
