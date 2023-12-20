abstract type PTBuilder end

Algebra.LoopStyle(::Type{<:PTBuilder}) = Loop()

struct SparsePTMatrixBuilder{B} <: PTBuilder
  builder::B
  length::Integer
end

struct PTArrayBuilder{B} <: PTBuilder
  builder::B
  length::Integer
end

struct PTCounter{C}
  counter::C
  length::Integer
end

Algebra.LoopStyle(::Type{<:PTCounter}) = Loop()

struct PTInserter{I}
  inserters::PTArray{I}
  function PTInserter(inserter::I,len::Integer) where I
    array = Vector{I}(undef,len)
    for i = 1:len
      array[i] = copy(inserter)
    end
    new{I}(PTArray(array))
  end
end

Algebra.LoopStyle(::Type{<:PTInserter}) = Loop()

function PTSparseMatrixAssembler(assem::SparseMatrixAssembler,μ,t)
  len = _length(μ,t)
  GenericSparseMatrixAssembler(
    SparsePTMatrixBuilder(assem.matrix_builder,len),
    PTArrayBuilder(assem.vector_builder,len),
    assem.rows,
    assem.cols,
    assem.strategy)
end

function Algebra.nz_counter(b::PTBuilder,args...)
  counter = nz_counter(b.builder,args...)
  PTCounter(counter,b.length)
end

function Algebra.nz_allocation(c::PTCounter)
  inserter = nz_allocation(c.counter)
  PTInserter(inserter,c.length)
end

@inline function Algebra.add_entry!(f::Function,c::PTCounter,args...)
  add_entry!(f,c.counter,args...)
end

function Algebra.create_from_nz(i::PTInserter)
  A = create_from_nz(first(i.inserters))
  array = Vector{typeof(A)}(undef,length(i.inserters))
  for j = eachindex(i.inserters)
    array[j] = copy(A)
  end
  PTArray(array)
end

function Base.copy(i::Algebra.InserterCSC)
  Algebra.InserterCSC(
    copy(i.nrows),
    copy(i.ncols),
    copy(i.colptr),
    copy(i.colnnz),
    copy(i.rowval),
    copy(i.nzval))
end

@inline function Algebra.add_entry!(f::Function,i::PTInserter,args...)
  for inserter in i.inserters
    add_entry!(f,inserter,args...)
  end
end

@inline function Algebra.add_entry!(f::Function,i::PTInserter,v::PTArray,args...)
  for (inserter,value) in zip(i.inserters,v.array)
    add_entry!(f,inserter,value,args...)
  end
end

function Algebra.nz_allocation(c::PTCounter{<:Algebra.CounterCOO})
  allocation = nz_allocation(c.counter)
  PTAllocationCOO(allocation,c.length)
end

struct PTAllocationCOO{T,A,B,C}
  allocation::Algebra.AllocationCOO{T,A,B,C}
  length::Integer
end

Algebra.LoopStyle(::Type{<:PTAllocationCOO}) = Loop()

@inline function Algebra.add_entry!(f::Function,a::PTAllocationCOO,args...)
  add_entry!(f,a.allocation,args...)
end

@inline function Algebra.add_entry!(f::Function,a::PTAllocationCOO,v::PTArray,args...)
  @notimplemented
end

function Algebra.create_from_nz(a::PTAllocationCOO)
  A = create_from_nz(a.allocation)
  array = Vector{typeof(A)}(undef,a.length)
  for j = 1:a.length
    array[j] = copy(A)
  end
  PTArray(array)
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

Algebra.create_from_nz(a::PTArray) = a

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
  vs::PTArray,
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
  vs::PTArray,
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
