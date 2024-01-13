function Algebra.allocate_vector(
  a::SparseMatrixAssember,
  vecdata::Tuple{<:AbstractVector{<:PArray},Any})

  cellvec,cellidsrows = vecdata
  cellvec1 = first.(cellvec)
  n = length(first(cellvec))
  vec = allocate_vector(a,(cellvec1,cellidsrows))
  parray(vec,n)
end

function Algebra.allocate_matrix(
  a::SparseMatrixAssember,
  matdata::Tuple{<:AbstractVector{<:PArray},Any,Any})

  cellmat,cellidsrows,cellidscols = matdata
  cellmat1 = first.(cellmat)
  n = length(first(cellmat))
  mat = allocate_matrix(a,(cellmat1,cellidsrows,cellidscols))
  parray(mat,n)
end

function FESpaces.collect_cell_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  w = []
  r = []
  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  push!(w,cell_vec_r)
  push!(r,rows)
  (w,r)
end

function FESpaces.collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  w = []
  r = []
  c = []
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
  (w,r,c)
end

Algebra.create_from_nz(a::PArray) = a

@inline function Algebra._add_entries!(
  combine::Function,
  A::AbstractMatrix{T},
  vs,
  is,js) where T<:AbstractArray

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          for Ak in A
            vij = vs[li,lj]
            add_entry!(combine,Ak,vij,i,j)
          end
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,
  A::AbstractMatrix{T},
  vs::Nothing,
  is,js) where T<:AbstractArray

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          for Ak in A
            add_entry!(combine,Ak,nothing,i,j)
          end
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,
  A::AbstractMatrix{T},
  vs::PArray,
  is,js) where T<:AbstractArray

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          for (Ak,vsk) in zip(A,vs)
            vijk = vsk[li,lj]
            add_entry!(combine,Ak,vijk,i,j)
          end
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,
  A::AbstractVector{T},
  vs::Nothing,
  is) where T<:AbstractArray

  for (li,i) in enumerate(is)
    if i>0
      for Ak in A
        add_entry!(Ak,nothing,i)
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,
  A::AbstractVector{T},
  vs,
  is) where T<:AbstractArray

  for (li,i) in enumerate(is)
    if i>0
      for Ak in A
        vi = vs[li]
        add_entry!(Ak,vi,i)
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,
  A::AbstractVector{T},
  vs::PArray,
  is) where T<:AbstractArray

  for (li,i) in enumerate(is)
    if i>0
      for (Ak,vsk) in zip(A,vs)
        vik = vsk[li]
        add_entry!(Ak,vik,i)
      end
    end
  end
  A
end
