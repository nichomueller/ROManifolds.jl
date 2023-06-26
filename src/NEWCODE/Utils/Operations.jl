function projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v) : v*sum(vnew'*X*v)
  proj_mat = reshape(similar(basis),:,1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v)/sum(v'*v) : v*sum(vnew'*X*v)/sum(v'*X*v)
  proj_mat = reshape(similar(basis),:,1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_complement(
  v::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  kwargs...)

  compl = reshape(similar(basis),:,1)
  copyto!(compl,v - orth_projection(v,basis;kwargs...))
end

function gram_schmidt(
  mat::AbstractMatrix{Float},
  basis::AbstractMatrix{Float};
  kwargs...)

  for i = axes(mat,2)
    mat_i = mat[:,i]
    mat_i = orth_complement(mat_i,basis;kwargs...)
    if i > 1
      mat_i = orth_complement(mat_i,mat[:,1:i-1];kwargs...)
    end
    mat[:,i] = mat_i/norm(mat_i)
  end

  mat
end

function expand(tup::Tuple)
  ntup = ()
  for el = tup
    if typeof(el) <: Tuple
      ntup = (ntup...,expand(el)...)
    else
      ntup = (ntup...,el)
    end
  end
  ntup
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

function Gridap.FESpaces.allocate_matrix(::EMatrix{T},sizes...) where T
  Elemental.zeros(EMatrix{T},sizes...)
end

function Gridap.FESpaces.allocate_matrix(::Matrix{T},sizes...) where T
  zeros(T,sizes...)
end

function Base.getindex(
  emat::EMatrix{T},
  idx::Union{UnitRange{Int},Colon},
  k::Int) where T

  reshape(convert(Matrix{T},emat[idx,k:k]),:)
end

function Base.getindex(
  emat::EMatrix{T},
  k::Int,
  idx::Union{UnitRange{Int},Colon}) where T

  reshape(convert(Matrix{T},emat[k:k,idx]),:)
end

Gridap.FESpaces.get_cell_dof_ids(trian::Triangulation) = trian.grid.cell_node_ids

function collect_trian(a::DomainContribution)
  t = ()
  for strian in get_domains(a)
    t = (t...,strian)
  end
  t
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
      cell_mat, trian = move_contributions(scell_mat,strian)
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
    cell_vec, trian = move_contributions(scell_vec,strian)
    @assert ndims(eltype(cell_vec)) == 1
    cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
    rows = get_cell_dof_ids(test,trian)
    push!(w,cell_vec_r)
    push!(r,rows)
    end
  end
  (w,r)
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
