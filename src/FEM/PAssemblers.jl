function Algebra.allocate_vector(
  ::Type{PArray{T,N,A,L}},
  n::Integer) where {T,N,A,L}

  S = eltype(T)
  vector = zeros(S,n)
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

function get_passembler(a::SparseMatrixAssembler,r::Union{PRealization,TransientPRealization})
  vec = get_vector_builder(a)
  vector_type = get_array_type(vec)
  pvector_type = typeof(PArray{vector_type}(undef,length(r)))
  mat = get_matrix_builder(a)
  matrix_type = get_array_type(mat)
  pmatrix_type = typeof(PArray{matrix_type}(undef,length(r)))
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  SparseMatrixAssembler(
    SparseMatrixBuilder(pmatrix_type),
    ArrayBuilder(pvector_type),
    rows,cols,strategy)
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::PFESpace,
  test::FESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy())

  N = length_free_values(trial)
  pmat = typeof(PArray{mat}(undef,N))
  pvec = typeof(PArray{vec}(undef,N))
  rows = get_free_dof_ids(test)
  cols = get_free_dof_ids(trial)
  GenericSparseMatrixAssembler(
    SparseMatrixBuilder(pmat),
    ArrayBuilder(pvec),
    rows,
    cols,
    strategy)
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::MultiFieldPFESpace{MS},
  test::MultiFieldFESpace{MS},
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
  ) where MS <: BlockMultiFieldStyle

  println(mat)
  println(vec)
  N = length_free_values(trial)
  pmat = typeof(PArray{mat}(undef,N))
  pvec = typeof(PArray{vec}(undef,N))
  mfs = MultiFieldStyle(test)
  MultiField.BlockSparseMatrixAssembler(
    mfs,
    trial,
    test,
    SparseMatrixBuilder(pmat),
    ArrayBuilder(pvec),
    strategy)
end

function FESpaces.SparseMatrixBuilder(::Type{PArray{T,N,A,L}},args...) where {T,N,A,L}
  builders = map(1:L) do i
    SparseMatrixBuilder(T,args...)
  end
  PArray(builders)
end

function FESpaces.ArrayBuilder(::Type{PArray{T,N,A,L}},args...) where {T,N,A,L}
  builders = map(1:L) do i
    ArrayBuilder(T,args...)
  end
  PArray(builders)
end

Algebra.LoopStyle(a::Type{PArray{T,N,A,L}}) where {T,N,A,L} = LoopStyle(T)

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

@inline function Algebra._add_entries!(combine::Function,A::PArray,vs::Nothing,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          map(A) do A
            add_entry!(combine,A,nothing,i,j)
          end
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(combine::Function,A::PArray,vs,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          map(A) do A
            vij = vs[li,lj]
            add_entry!(combine,A,vij,i,j)
          end
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(combine::Function,A::PArray,vs::PArray,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          map(A,vs) do A,vs
            vij = vs[li,lj]
            add_entry!(combine,A,vij,i,j)
          end
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(combine::Function,A::PArray,vs::Nothing,is)
  for (li,i) in enumerate(is)
    if i>0
      map(A) do A
        add_entry!(A,nothing,i)
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(combine::Function,A::PArray,vs,is)
  for (li,i) in enumerate(is)
    if i>0
      map(A) do A
        vi = vs[li]
        add_entry!(A,vi,i)
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(combine::Function,A::PArray,vs::PArray,is)
  for (li,i) in enumerate(is)
    if i>0
      map(A,vs) do A,vs
        vi = vs[li]
        add_entry!(A,vi,i)
      end
    end
  end
  A
end
