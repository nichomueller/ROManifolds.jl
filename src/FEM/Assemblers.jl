#function Gridap.assemble_matrix(f,U::Function,V::FESpace)
function my_assemble_matrix(f,U::Function,V::FESpace,times)
  a = SparseMatrixAssembler(U(0.),V)
  my_assemble_matrix(f,a,U,V,times)
end

#function Gridap.assemble_matrix(f::Function,a::Assembler,U::Function,V::FESpace)
function my_assemble_matrix(f::Function,a::Assembler,U::Function,V::FESpace,times)
  v = get_fe_basis(V)
  u(t) = get_trial_fe_basis(U(t))
  cm(t) = collect_cell_matrix(U(t),V,f(t,u(t),v))

  mfixed = fixed_assemble_matrix(a,cm(0.))
  mnew = Broadcasting(t->new_assemble_matrix(a,mfixed,cm,t))(times)
  mnew
end

#function Gridap.assemble_matrix(a::SparseMatrixAssembler,matdata)
function fixed_assemble_matrix(a::SparseMatrixAssembler,matdata)
  m1 = Gridap.FESpaces.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  symbolic_loop_matrix!(m1,a,matdata)
  m2 = Gridap.FESpaces.nz_allocation(m1)
  numeric_loop_matrix!(m2,a,matdata)
  m3 = fixed_from_nz(m2)
  m3
end

function fixed_from_nz(a::Gridap.Algebra.InserterCSC{Tv,Ti}) where {Tv,Ti}
  k = 1
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      a.nzval[k] = a.nzval[p]
      k += 1
    end
  end
  @inbounds for j in 1:a.ncols
    a.colptr[j+1] = a.colnnz[j]
  end
  length_to_ptrs!(a.colptr)
  nnz = a.colptr[end]-1
  resize!(a.rowval,nnz)
  a
end

function new_assemble_matrix(a::SparseMatrixAssembler,mfixed,matdata,t)
  m1 = Gridap.FESpaces.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  matdata_t = matdata(t)
  symbolic_loop_matrix!(m1,a,matdata_t)
  m2 = Gridap.FESpaces.nz_allocation(m1)
  numeric_loop_matrix!(m2,a,matdata_t)
  m3 = new_from_nz(m2,mfixed)
  m3
end

#function Gridap.FESpaces.create_from_nz(a::Gridap.Algebra.InserterCSC{Tv,Ti}) where {Tv,Ti}
function new_from_nz(
  a::Gridap.Algebra.InserterCSC{Tv,Ti},
  af::Gridap.Algebra.InserterCSC{Tv,Ti}) where {Tv,Ti}

  k = 1
  for j in 1:af.ncols
    pini = Int(af.colptr[j])
    pend = pini + Int(af.colnnz[j]) - 1
    for p in pini:pend
      a.nzval[k] = a.nzval[p]
      k += 1
    end
  end
  resize!(a.nzval,af.colptr[end]-1)
  SparseMatrixCSC(af.nrows,af.ncols,af.colptr,af.rowval,a.nzval)
end



dΩ = measures.dΩ
f(t,u,v) = ∫(sin(t)*∇(v)⋅∇(u))dΩ
g(x,t::Real) = sin(t)
g(t::Real) = x->g(x,t)
reffe = Gridap.ReferenceFE(lagrangian,Float,1)
V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
U(t) = TrialFESpace(V,g(t))
times = collect(1:1:10)
A = my_assemble_matrix(f,U,V,times)
