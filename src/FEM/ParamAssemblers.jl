function Algebra.allocate_vector(::Type{V},n::Integer) where V<:AbstractParamContainer
  vector = zeros(eltype(V),n)
  allocate_param_array(vector,length(V))
end

function Algebra.allocate_in_range(matrix::ParamMatrix{T,A,L}) where {T,A,L}
  V = ParamVector{T,Vector{eltype(A)},L}
  allocate_in_range(V,matrix)
end

function Algebra.allocate_in_domain(matrix::ParamMatrix{T,A,L}) where {T,A,L}
  V = ParamVector{T,Vector{eltype(A)},L}
  allocate_in_domain(V,matrix)
end

function get_param_matrix_builder(
  a::SparseMatrixAssembler,
  r::Union{ParamRealization,TransientParamRealization})

  L = length(r)
  mat = get_matrix_builder(a)
  M = get_array_type(mat)
  elM = eltype(M)
  pmatrix_type = ParamMatrix{elM,Vector{M},L}
  SparseMatrixBuilder(pmatrix_type)
end

function get_param_vector_builder(
  a::SparseMatrixAssembler,
  r::Union{ParamRealization,TransientParamRealization})

  L = length(r)
  vec = get_vector_builder(a)
  V = get_array_type(vec)
  elV = eltype(V)
  pvector_type = ParamVector{elV,Vector{V},L}
  ArrayBuilder(pvector_type)
end

function get_param_assembler(
  a::SparseMatrixAssembler,
  r::Union{ParamRealization,TransientParamRealization})

  matrix_builder = get_param_matrix_builder(a,r)
  vector_builder = get_param_vector_builder(a,r)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  GenericSparseMatrixAssembler(matrix_builder,vector_builder,rows,cols,strategy)
end

function FESpaces.SparseMatrixAssembler(
  mat,
  vec,
  trial::ParamFESpace,
  test::FESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy())

  N = length_free_values(trial)
  pmat = typeof(ParamMatrix{mat}(undef,N))
  pvec = typeof(ParamVector{vec}(undef,N))
  rows = get_free_dof_ids(test)
  cols = get_free_dof_ids(trial)
  GenericSparseMatrixAssembler(
    SparseMatrixBuilder(pmat),
    ArrayBuilder(pvec),
    rows,
    cols,
    strategy)
end

function FESpaces.SparseMatrixBuilder(::Type{ParamArray{T,2,A,L}} where T,args...) where {A,L}
  builder = map(1:L) do i
    SparseMatrixBuilder(eltype(A),args...)
  end
  ParamContainer(builder)
end

function FESpaces.ArrayBuilder(::Type{ParamArray{T,1,A,L}} where T,args...) where {A,L}
  builder = map(1:L) do i
    ArrayBuilder(eltype(A),args...)
  end
  ParamContainer(builder)
end

Algebra.LoopStyle(a::Type{<:ParamContainer{T,L}}) where {T,L} = LoopStyle(T)

function Algebra.nz_counter(builder::ParamContainer,axes)
  counter = map(builder) do builder
    Algebra.nz_counter(builder,axes)
  end
  ParamContainer(counter)
end

function Algebra.nz_allocation(a::ParamContainer)
  inserter = map(a) do a
    Algebra.nz_allocation(a)
  end
  ParamContainer(inserter)
end

function Algebra.create_from_nz(a::ParamContainer)
  array = map(a) do a
    Algebra.create_from_nz(a)
  end
  ParamArray(array)
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
  combine::Function,A::AbstractParamContainer,vs::Nothing,is,js)
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

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs,is,js)
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

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::AbstractParamContainer,is,js)
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

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::Nothing,is)
  for (li,i) in enumerate(is)
    if i>0
      map(A) do A
        add_entry!(A,nothing,i)
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs,is)
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

@inline function Algebra._add_entries!(
  combine::Function,A::AbstractParamContainer,vs::AbstractParamContainer,is)
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

function test_passembler(a::Assembler,matdata,vecdata,data)
  A = allocate_matrix(a,matdata)
  @test FESpaces.num_cols(a) == size(A,2)
  @test FESpaces.num_rows(a) == size(A,1)
  assemble_matrix!(A,a,matdata)
  assemble_matrix_add!(A,a,matdata)
  A = assemble_matrix(a,matdata)
  @test FESpaces.num_cols(a) == size(A,2)
  @test FESpaces.num_rows(a) == size(A,1)
  b = allocate_vector(a,vecdata)
  @test length(testitem(b)) == FESpaces.num_rows(a)
  assemble_vector!(b,a,vecdata)
  assemble_vector_add!(b,a,vecdata)
  b = assemble_vector(a,vecdata)
  @test length(testitem(b)) == FESpaces.num_rows(a)
  A, b = allocate_matrix_and_vector(a,data)
  assemble_matrix_and_vector!(A,b,a,data)
  assemble_matrix_and_vector_add!(A,b,a,data)
  @test FESpaces.num_cols(a) == size(A,2)
  @test FESpaces.num_rows(a) == size(A,1)
  @test length(testitem(b)) == FESpaces.num_rows(a)
  A, b = assemble_matrix_and_vector(a,data)
  @test FESpaces.num_cols(a) == size(A,2)
  @test FESpaces.num_rows(a) == size(A,1)
  @test length(testitem(b)) == FESpaces.num_rows(a)
end
