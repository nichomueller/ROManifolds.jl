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
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces

domain =(0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

params = [1],[2],[3]
μ = ParamRealization([[1],[2],[3]])
μ₀ = ParamRealization([[0],[0],[0]])
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
U = TrialParamFESpace(V,fμ(μ₀))

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

T = SparseMatrixCSC{Float64,Int}

matvecdata = ( term_to_cellmatvec , term_to_rows, term_to_cols)
matdata = (term_to_cellmat,term_to_rows,term_to_cols)
vecdata = (term_to_cellvec,term_to_rows)
data = (matvecdata,matdata,vecdata)

assem = ParamFESpaces.get_param_assembler(SparseMatrixAssembler(T,Vector{Float64},U,V),μ)
test_sparse_matrix_assembler(assem,matdata,vecdata,data)

strategy = GenericAssemblyStrategy(row->row,col->col,row->true,col->true)

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
@test typeof(x) == typeof(x2) == typeof(vec) <: ParamVector

for (i,μi) = enumerate(μ)
  μi = sum(μi)
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
  μi = sum(μi)
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
  μi = sum(μi)
  @test vec[i] ≈ [0.0625, 0.125, 0.0625]*μi
  @test mat[i][1, 1]  ≈  1.333333333333333*μi
  @test mat[i][2, 1]  ≈ -0.33333333333333*μi
end

mat, vec = assemble_matrix_and_vector(assem2,data)

x4 = mat \ vec
@test x ≈ x4

for (i,μi) = enumerate(μ)
  μi = sum(μi)
  @test vec[i] ≈ [0.0625, 0.125, 0.0625]*μi
  @test mat[i][1, 1]  ≈  1.333333333333333*μi
  @test mat[i][2, 1]  ≈ -0.33333333333333*μi
end

################################################################################

dtrian = Measure(trian,2)
biform(u,v) = ∫(fμ(μ)*∇(v)⊙∇(u))dtrian
A = assemble_matrix(biform,U,V)

# A = assemble_matrix(biform,U,V)
a = SparseMatrixAssembler(U,V)
dc = biform(get_trial_fe_basis(U),get_fe_basis(V))
matdata = collect_cell_matrix(U,V,dc)



# ok
bf(u,v) = ∫(∇(v)⊙∇(u))dtrian
AA = assemble_matrix(bf,V,V)

aa = SparseMatrixAssembler(V,V)
dcc = bf(get_trial_fe_basis(V),get_fe_basis(V))
mdata = collect_cell_matrix(V,V,dcc)






















lazy_getter(a,i=1) = lazy_map(x->getindex(x.data,i),a)

domain = (0,1,0,1)
cells = (2,2)
model = simplexify(CartesianDiscreteModel(domain,cells))

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)
Λ = SkeletonTriangulation(model)
n_Λ = get_normal_vector(Λ)

degree = 2
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΛ = Measure(Λ,degree)

v = GenericCellField(get_cell_shapefuns(Ω),Ω,ReferenceDomain())
u = GenericCellField(lazy_map(transpose,get_data(v)),Ω,ReferenceDomain())

μ = ParamRealization([[1],[2],[3]])
μ₀ = ParamRealization([[0],[0],[0]])
f(x,μ) = 1+sum(μ)
f(μ) = x -> f(x,μ)
fμfμ = 𝑓ₚ(f,μ)
fμ₀ = 𝑓ₚ(f,μ₀)

a = ∫(fμfμ*u*v)*dΩ + ∫(fμfμ*u*v)*dΓ + ∫(fμfμ*∇(u)⋅∇(v))*dΩ
@test num_domains(a) == 2
@test Ω in get_domains(a)
@test Γ in get_domains(a)
@test isa(get_contribution(a,Ω),LazyArray{<:Any,<:AbstractParamArray})
@test length(lazy_getter(get_contribution(a,Ω))) == num_cells(Ω)
@test param_length(first(get_contribution(a,Ω))) == 3
bb = ∫(u*v)*dΩ + ∫(u*v)*dΓ + ∫(∇(u)⋅∇(v))*dΩ
@test 2*sum(bb) == sum(a)[1]

#################
boh = sum(a[Ω])
bohok = sum(bb[Ω])

aa = a[Ω]
cache = array_cache(aa)
r = zero(testitem(aa))
for i in eachindex(aa)
  ai = getindex!(cache,aa,i)
  r += ai
end

cc = bb[Ω]
ccache = array_cache(cc)
rr = zero(testitem(cc))
for i in eachindex(cc)
  ci = getindex!(ccache,cc,i)
  rr += ci
end

for i in eachindex(aa)
  println(i)
  ai = getindex!(cache,aa,i)
  ci = getindex!(ccache,cc,i)
  @test ai[1] ≈ 2*ci
end

x = get_cell_points(Ω)
ceval = (fμfμ*u*v)(x)

data = param_data(ceval[1])

ceval2 = 2*(u*v)(x)
ceval3 = 3*(u*v)(x)
ceval4 = 4*(u*v)(x)

data234 = [ceval2[1],ceval3[1],ceval4[1]]
#################

a = ∫(fμ₀)*dΩ + ∫(fμ₀)*dΓ
@test all(sum(a) .≈ 5)
@test all(sum(2*a) .≈ 10)
@test all(sum(a*2) .≈ 10)

f1(x,μ) = 2*x[1]*(1+sum(μ))
f1(μ) = x -> f1(x,μ)
f1μ = 𝑓ₚ(f1,μ₀)
f2(x,μ) = 3*x[2]*(1+sum(μ))
f2(μ) = x -> f2(x,μ)
f2μ = 𝑓ₚ(f2,μ₀)
u = CellField(f1μ,Ω)
v = CellField(f2μ,Ω)

a = ∫(jump(u))*dΛ
@test all(sum(a) .+ 1 .≈ 1)

a = ∫( (n_Λ.⁺⋅∇(v.⁻))*jump(n_Λ⋅∇(u)) )*dΛ
@test all(sum(a) .+ 1 .≈ 1)

quad = Quadrature(duffy,2)
dΩ = Measure(Ω,quad)
s = ∫(fμ₀)dΩ
@test all(sum(s) .≈ 1)

dΩ = Measure(Ω,degree,T=Float32)
dΓ = Measure(Γ,degree,T=Float32)
dΛ = Measure(Λ,degree,T=Float32)

a = ∫(fμ₀)*dΩ + ∫(fμ₀)*dΓ
@test all(isapprox.(sum(a),5,atol=1e-6))
@test all(isapprox.(sum(2*a),10,atol=1e-6))
@test all(isapprox.(sum(a*2),10,atol=1e-6))

u = CellField(f1μ,Ω)
v = CellField(f2μ,Ω)

a = ∫(jump(u))*dΛ
@test all(sum(a) .+ 1 .≈ 1)

a = ∫( (n_Λ.⁺⋅∇(v.⁻))*jump(n_Λ⋅∇(u)) )*dΛ
@test all(sum(a) .+ 1 .≈ 1)

quad = Quadrature(duffy,2,T=Float32)
dΩ = Measure(Ω,quad)
s = ∫(fμ₀)dΩ
@test all(sum(s) .≈ 1)
