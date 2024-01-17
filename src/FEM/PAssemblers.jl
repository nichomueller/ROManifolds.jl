function Algebra.allocate_vector(
  ::Type{PArray{T,N,A,L}},
  n::Integer) where {T,N,A,L}

  vector = zeros(T,n)
  allocate_parray(vector,L)
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

function Algebra.allocate_in_domain(matrix::PArray{<:AbstractMatrix})
  map(matrix) do matrix
    allocate_in_domain(matrix)
  end
end

struct SparseMatrixPAssembler{M<:PArray,V<:PArray} <: SparseMatrixAssembler
  matrix_builder::M
  vector_builder::V
  rows::AbstractUnitRange
  cols::AbstractUnitRange
  strategy::AssemblyStrategy
end

function SparseMatrixPAssembler(
  mat,vec,
  trial::PFESpace,
  test::FESpace,
  strategy::AssemblyStrategy=FESpaces.DefaultAssemblyStrategy())

  rows = get_free_dof_ids(test)
  cols = get_free_dof_ids(trial)
  SparseMatrixPAssembler(
    SparseMatrixBuilder(mat),
    ArrayBuilder(vec),
    rows,
    cols,
    strategy)
end

function FESpaces.SparseMatrixAssembler(
  mat,vec,
  trial::PFESpace,
  test::FESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy())

  SparseMatrixPAssembler(mat,vec,trial,test,strategy)
end

function FESpaces.SparseMatrixAssembler(mat,trial::PFESpace,test::FESpace)
  mat_builder = SparseMatrixBuilder(mat)
  T = eltype(get_array_type(mat_builder))
  N = length_free_values(trial)
  vector_type = Vector{T}
  pvector_type = PArray{vector_type}(undef,N)
  SparseMatrixAssembler(mat_builder,pvector_type,trial,test)
end

function FESpaces.SparseMatrixAssembler(trial::PFESpace,test::FESpace)
  T = get_dof_value_type(trial)
  N = length_free_values(trial)
  matrix_type = SparseMatrixCSC{T,Int}
  pmatrix_type = PArray{matrix_type}(undef,N)
  vector_type = Vector{T}
  pvector_type = PArray{vector_type}(undef,N)
  SparseMatrixAssembler(matrix_type,vector_type,trial,test)
end

FESpaces.get_rows(a::SparseMatrixPAssembler) = a.rows

FESpaces.get_cols(a::SparseMatrixPAssembler) = a.cols

FESpaces.get_matrix_builder(a::SparseMatrixPAssembler) = a.matrix_builder

FESpaces.get_vector_builder(a::SparseMatrixPAssembler) = a.vector_builder

FESpaces.get_assembly_strategy(a::SparseMatrixPAssembler) = a.strategy

function Algebra.nz_counter(builder::PArray,axes)
  map(builder) do builder
    Algebra.nz_counter(builder,axes)
  end
end

function Algebra.nz_allocation(a::PArray)
  map(a) do a
    Algebra.nz_allocation(a)
  end
end

function Algebra.create_from_nz(a::PArray)
  map(a) do a
    Algebra.create_from_nz(a)
  end
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
