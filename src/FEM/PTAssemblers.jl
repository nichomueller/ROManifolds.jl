struct PTSparseMatrixAssember <: Assembler
  assem::SparseMatrixAssembler
end

function PTSparseMatrixAssember(trial::TrialPFESpace,test::FESpace)
  trial0 = trial(nothing)
  PTSparseMatrixAssember(PTSparseMatrixAssember(trial0,test))
end

function PTSparseMatrixAssember(trial::TransientTrialPFESpace,test::FESpace)
  trial0 = trial(nothing,nothing)
  PTSparseMatrixAssember(PTSparseMatrixAssember(trial0,test))
end

function Algebra.allocate_vector(a::PTSparseMatrixAssember,vecdata)

  data = first()
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
