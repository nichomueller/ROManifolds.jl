# module SparseMatrixAssemblers

using Test
using Gridap.Arrays
using Gridap.TensorValues
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.Algebra
using SparseArrays
using SparseMatricesCSR
using Gridap.FESpaces
using Gridap.CellData
using Gridap.Algebra
using Mabla.FEM

domain =(0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

params = [1],[2],[3]
μ = PRealization([[1],[2],[3]])
μ₀ = PRealization([[0],[0],[0]])
f(x,μ) = sum(μ)
f(μ) = x -> f(x,μ)
fμ(μ) = 𝑓ₚ(f,μ)
b(x) = x[2]
biform(u,v) = fμ(μ)*∇(v)⊙∇(u)
liform(v) = fμ(μ)*v⊙b
biform1(u,v) = fμ(μ)*v*u
liform1(v) = fμ(μ)*v*3

reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(model,reffe,dirichlet_tags=[1,2,3,4,6,5])
U = TrialPFESpace(V,fμ(μ₀))

v = get_fe_basis(V)
u = get_trial_fe_basis(U)

degree = 2
trian = get_triangulation(model)
quad = CellQuadrature(trian,degree)

btrian = BoundaryTriangulation(model)
bquad = CellQuadrature(btrian,degree)

b0trian = Triangulation(model,Int[])
b0quad = CellQuadrature(b0trian,degree)

cellmat = integrate(biform(u,v),quad)
cellvec = integrate(liform(v),quad)
cellmatvec = pair_arrays(cellmat,cellvec)
rows = get_cell_dof_ids(V,trian)
cols = get_cell_dof_ids(U,trian)
cellmat_c = attach_constraints_cols(U,cellmat,trian)
cellmat_rc = attach_constraints_rows(V,cellmat_c,trian)
cellvec_r = attach_constraints_rows(V,cellvec,trian)
cellmatvec_c = attach_constraints_cols(U,cellmatvec,trian)
cellmatvec_rc = attach_constraints_rows(V,cellmatvec_c,trian)

bcellmat = integrate(biform1(u,v),bquad)
bcellvec = integrate(liform1(v),bquad)
bcellmatvec = pair_arrays(bcellmat,bcellvec)
brows = get_cell_dof_ids(V,btrian)
bcols = get_cell_dof_ids(U,btrian)
bcellmat_c = attach_constraints_cols(U,bcellmat,btrian)
bcellmat_rc = attach_constraints_rows(V,bcellmat_c,btrian)
bcellvec_r = attach_constraints_rows(V,bcellvec,btrian)
bcellmatvec_c = attach_constraints_cols(U,bcellmatvec,btrian)
bcellmatvec_rc = attach_constraints_rows(V,bcellmatvec_c,btrian)

b0cellmat = integrate(biform1(u,v),b0quad)
b0cellvec = integrate(liform1(v),b0quad)
b0cellmatvec = pair_arrays(b0cellmat,b0cellvec)
b0rows = get_cell_dof_ids(V,b0trian)
b0cols = get_cell_dof_ids(U,b0trian)
@test length(b0rows) == 0
b0cellmat_c = attach_constraints_cols(U,b0cellmat,b0trian)
b0cellmat_rc = attach_constraints_rows(V,b0cellmat_c,b0trian)
b0cellvec_r = attach_constraints_rows(V,b0cellvec,b0trian)
b0cellmatvec_c = attach_constraints_cols(U,b0cellmatvec,b0trian)
b0cellmatvec_rc = attach_constraints_rows(V,b0cellmatvec_c,b0trian)

term_to_cellmat = [cellmat_rc, bcellmat_rc, b0cellmat_rc]
term_to_cellvec = [cellvec, bcellvec, b0cellvec]
term_to_rows = [rows, brows, b0rows]
term_to_cols = [cols, bcols, b0cols]
term_to_cellmatvec = [ cellmatvec, bcellmatvec, b0cellmatvec ]

mtypes = [
  SparseMatrixCSC{Float64,Int},
  SparseMatrixCSR{0,Float64,Int},
  SparseMatrixCSR{1,Float64,Int},
  SymSparseMatrixCSR{0,Float64,Int},
  SymSparseMatrixCSR{1,Float64,Int}]

# for T in mtypes
  T = SparseMatrixCSC{Float64,Int}
  pmatrix_type = typeof(PArray{T}(undef,3))
  pvector_type = typeof(PArray{Vector{Float64}}(undef,3))

  matvecdata = ( term_to_cellmatvec , term_to_rows, term_to_cols)
  matdata = (term_to_cellmat,term_to_rows,term_to_cols)
  vecdata = (term_to_cellvec,term_to_rows)
  data = (matvecdata,matdata,vecdata)

  assem = SparseMatrixAssembler(pmatrix_type,pvector_type,U,V)
  @test isa(assem,SparseMatrixPAssembler)
  test_sparse_matrix_assembler(assem,matdata,vecdata,data)

  @inline function Algebra._add_entries!(combine::Function,A,vs::Nothing,is,js)
    println(typeof(A))
    for (lj,j) in enumerate(js)
      if j>0
        for (li,i) in enumerate(is)
          if i>0
            add_entry!(combine,A,nothing,i,j)
          end
        end
      end
    end
    A
  end

  # A = allocate_matrix(assem,matdata)
  m1 = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
  symbolic_loop_matrix!(m1,assem,matdata)
  m2 = nz_allocation(m1)
  symbolic_loop_matrix!(m2,assem,matdata)
  m3 = create_from_nz(m2)

  getter(a) = map(first,a.array)
  lazy_getter(a) = lazy_map(getter,a)
  A = SparseMatrixAssembler(T,Vector{Float64},V,V)
  terms = [lazy_getter(cellmat_rc), lazy_getter(bcellmat_rc), b0cellmat_rc]
  mdata = (terms,term_to_rows,term_to_cols)

  n1 = nz_counter(get_matrix_builder(A),(get_rows(A),get_cols(A)))
  symbolic_loop_matrix!(n1,A,mdata)
  n2 = nz_allocation(n1)
  symbolic_loop_matrix!(n2,A,mdata)
  n3 = create_from_nz(n2)

  strategy = GenericAssemblyStrategy(row->row,col->col,row->true,col->true)

  assem2 = SparseMatrixAssembler(pmatrix_type,pvector_type,U,V,strategy)
  test_sparse_matrix_assembler(assem2,matdata,vecdata,data)

  matdata = ([cellmat],[rows],[cols])
  vecdata = ([cellvec],[rows])

  mat = assemble_matrix(assem,matdata)
  vec = assemble_vector(assem,vecdata)
  x = mat \ vec


  assemble_matrix!(mat,assem,matdata)
  assemble_vector!(vec,assem,vecdata)
  x2 = mat \ vec

  @test x ≈ x2
  @test length(x) == length(x2) == length(vec) == length(mat) == 3
  @test typeof(x) == typeof(x2) == typeof(vec) == pvector_type

  for (i,μi) = enumerate(μ)
    @test vec[i] ≈ [0.0625, 0.125, 0.0625]*μi
    @test mat[i][1, 1]  ≈  1.333333333333333*μi
    @test mat[i][2, 1]  ≈ -0.33333333333333*μi
    @test mat[i][1, 2]  ≈ -0.33333333333333*μi
    @test mat[i][2, 2]  ≈ 2.666666666666666*μi
    @test mat[i][3, 2]  ≈ -0.33333333333333*μi
    @test mat[i][2, 3]  ≈ -0.33333333333333*μi
    @test mat[i][3, 3]  ≈ 1.333333333333333*μi
  end

  data = (([cellmatvec],[rows],[cols]),([],[],[]),([],[]))
  mat, vec = allocate_matrix_and_vector(assem,data)
  assemble_matrix_and_vector!(mat,vec,assem,data)
  assemble_matrix_and_vector!(mat,vec,assem,data)

  for (i,μi) = enumerate(μ)
    @test vec[i] ≈ [0.0625, 0.125, 0.0625]*μi
    @test mat[i][1, 1]  ≈  1.333333333333333*μi
    @test mat[i][2, 1]  ≈ -0.33333333333333*μi
  end

  x3 = mat \ vec
  @test x ≈ x3

  mat, vec = assemble_matrix_and_vector(assem,data)

  x4 = mat \ vec
  @test x ≈ x4

  for (i,μi) = enumerate(μ)
    @test vec[i] ≈ [0.0625, 0.125, 0.0625]*μi
    @test mat[i][1, 1]  ≈  1.333333333333333*μi
    @test mat[i][2, 1]  ≈ -0.33333333333333*μi
  end

  mat, vec = assemble_matrix_and_vector(assem2,data)

  x4 = mat \ vec
  @test x ≈ x4

  for (i,μi) = enumerate(μ)
    @test vec[i] ≈ [0.0625, 0.125, 0.0625]*μi
    @test mat[i][1, 1]  ≈  1.333333333333333*μi
    @test mat[i][2, 1]  ≈ -0.33333333333333*μi
  end
# end

# end
