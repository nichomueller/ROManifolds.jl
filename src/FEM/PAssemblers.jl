function Algebra.allocate_vector(
  a::SparseMatrixAssembler,
  vecdata::Tuple{<:AbstractVector{<:AbstractVector{<:PArray}},Any})

  cellvec,cellidsrows = vecdata
  cellvec1 = first.(cellvec)
  n = length(first(cellvec))
  vec = allocate_vector(a,(cellvec1,cellidsrows))
  parray(vec,n)
end

function Algebra.allocate_matrix(
  a::SparseMatrixAssembler,
  matdata::Tuple{<:AbstractVector{<:AbstractVector{<:PArray}},Any,Any})

  cellmat,cellidsrows,cellidscols = matdata
  cellmat1 = first.(cellmat)
  n = length(first(cellmat))
  mat = allocate_matrix(a,(cellmat1,cellidsrows,cellidscols))
  parray(mat,n)
end

function FESpaces.collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution)

  map([get_domains(a)...]) do strian
    collect_cell_matrix_for_trian(trial,test,a,strian)
  end |> tuple_of_arrays
end

function collect_cell_matrix_for_trian(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(trial,trian)
  [cell_mat_rc],[rows],[cols]
end

function FESpaces.collect_cell_vector(test::FESpace,a::DomainContribution)
  map([get_domains(a)...]) do strian
    collect_cell_vector_for_trian(test,a,strian)
  end |> tuple_of_arrays
end

function collect_cell_vector_for_trian(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  [cell_vec_r],[rows]
end

function Algebra.allocate_in_range(matrix::PArray{<:AbstractMatrix})
  map(matrix) do matrix
    allocate_in_range(matrix)
  end
end

function Algebra.allocate_in_domain(::Type{<:PArray},matrix)
  map(matrix) do matrix
    allocate_in_domain(matrix)
  end
end

function Algebra.allocate_in_domain(matrix:::PArray{<:AbstractMatrix})
  map(matrix) do matrix
    allocate_in_domain(matrix)
  end
end

Algebra.create_from_nz(a::PArray) = a

@inline function Algebra._add_entries!(
  combine::Function,
  A::AbstractMatrix{<:AbstractArray},
  vs,
  is,js)

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
  A::AbstractMatrix{<:AbstractArray},
  vs::Nothing,
  is,js)

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
  A::AbstractMatrix{<:AbstractArray},
  vs::PArray,
  is,js)

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
  A::AbstractVector{<:AbstractArray},
  vs::Nothing,
  is)

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
  A::AbstractVector{<:AbstractArray},
  vs,
  is)

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
  A::AbstractVector{<:AbstractArray},
  vs::PArray,
  is)

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
