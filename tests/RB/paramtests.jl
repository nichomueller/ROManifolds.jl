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


##################
a = (∫(fμfμ*u*v)*dΩ + ∫(fμfμ*∇(u)⋅∇(v))*dΩ)[Ω] #+ ∫(fμfμ*u*v)*dΓ)#[Γ]
b = (∫(f([1])*u*v)*dΩ + ∫(f([1])*∇(u)⋅∇(v))*dΩ)[Ω] #+ ∫(f([1])*u*v)*dΓ)#[Γ]
x = get_cell_points(Ω)

for (ai,bi) in zip(a,b)
  @test param_getindex(ai,1) == bi
end

# cx = (fμfμ*u*v)(x)
f1 = fμfμ*u
ax1 = map(i->i(x),f1.args)
cache1 = return_cache(Fields.BroadcastingFieldOpMap(f1.op.op),ax1)
ax11 = first.(ax1)
cache1 = return_cache(Fields.BroadcastingFieldOpMap(f1.op.op),ax11...)
# res1 = evaluate!(cache1,Fields.BroadcastingFieldOpMap(f1.op.op),ax11...)

f11 = Fields.BroadcastingFieldOpMap(f1.op.op)
cache,pA,data = cache1
@inbounds for i in param_eachindex(data)
  data[i] = evaluate!(cache[i],f11,param_getindex.(pA,i)...)
end
data

# dx = (u*v)(x)
f2 = f([1])*u
ax2 = map(i->i(x),f2.args)
ax21 = first.(ax2)
cache2 = return_cache(Fields.BroadcastingFieldOpMap(f2.op.op),ax21...)
evaluate!(cache2,Fields.BroadcastingFieldOpMap(f2.op.op),ax21...)
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


cachea = array_cache(a)
r = zero(testitem(a))
for i in eachindex(a)
  ai = getindex!(cachea,a,i)
  r += ai
end

cacheb = array_cache(b)
_r = zero(testitem(b))
for i in eachindex(b)
  bi = getindex!(cacheb,b,i)
  _r += bi
end

for i = eachindex(a)
  @test getindex!(cachea,a,i)[1] == getindex!(cacheb,b,i)
end

i = (2,)
_cache1, index_and_item1 = cachea
index1 = LinearIndices(a)[i...]
# index_and_item.index != index
cg1, cgi1, cf1 = _cache1
gi1 = getindex!(cg1, a.maps, i...)
index_and_item1.item = Arrays._getindex_and_call!(cgi1,gi1,cf1,a.args,i...)
index_and_item1.index = index1

f11 = getindex!(cf1[1],a.args[1],i...)
f21 = getindex!(cf1[2],a.args[2],i...)

_cache11, index_and_item11 = cf1[2]
index11 = LinearIndices(a.args[2])[i...]
# index_and_item.index != index
cg11, cgi11, cf11 = _cache11
gi11 = getindex!(cg1, a.args[2].maps, i...)
index_and_item11.item = Arrays._getindex_and_call!(cgi11,gi11,cf11,a.args[2].args,i...)

f1 = getindex!(cf11[1],a.args[2].args[1],i...)
f2 = getindex!(cf11[2],a.args[2].args[2],i...)
f3 = getindex!(cf11[3],a.args[2].args[3],i...)
evaluate!(cgi11,gi11,f1,f2,f3)


_cache2, index_and_item2 = cacheb
index2 = LinearIndices(b)[i...]
# index_and_item.index != index
cg2, cgi2, cf2 = _cache2
gi2 = getindex!(cg2, b.maps, i...)
index_and_item2.item = Arrays._getindex_and_call!(cgi2,gi2,cf2,b.args,i...)
index_and_item2.index = index2

g11 = getindex!(cf2[1],b.args[1],i...)
g21 = getindex!(cf2[2],b.args[2],i...)

_cache22, index_and_item22 = cf2[2]
index22 = LinearIndices(b.args[2])[i...]
# index_and_item.index != index
cg22, cgi22, cf22 = _cache22
gi22 = getindex!(cg22, b.args[2].maps, i...)
index_and_item22.item = Arrays._getindex_and_call!(cgi22,gi22,cf22,b.args[2].args,i...)

h1 = getindex!(cf22[1],b.args[2].args[1],i...)
h2 = getindex!(cf22[2],b.args[2].args[2],i...)
h3 = getindex!(cf22[3],b.args[2].args[3],i...)
evaluate!(cgi22,gi22,h1,h2,h3)

for (ai,bi) in zip(a.args[2],b.args[2])
  @test ai[1] == bi
end

###########
# f1 = getindex!(cf11[1],f1.args[2].args[1],i...)
_cache, index_and_item = cf11[1]
index = LinearIndices(a.args[2].args[1])[i...]
cg, cgi, cf = _cache
gi = getindex!(cg, a.args[2].args[1].maps, i...)
index_and_item.item = Arrays._getindex_and_call!(cgi,gi,cf,a.args[2].args[1].args,i...)

f1 = getindex!(cf[1],a.args[2].args[1].args[1],i...)
f2 = getindex!(cf[2],a.args[2].args[1].args[2],i...)

# h1 = getindex!(cf22[1],b.args[2].args[1],i...)
_cache2, index_and_item2 = cf22[1]
index2 = LinearIndices(b.args[2].args[1])[i...]
cg2, cgi2, cf2 = _cache2
gi2 = getindex!(cg2, b.args[2].args[1].maps, i...)
index_and_item2.item = Arrays._getindex_and_call!(cgi2,gi2,cf2,b.args[2].args[1].args,i...)

g1 = getindex!(cf2[1],b.args[2].args[1].args[1],i...)
g2 = getindex!(cf2[2],b.args[2].args[1].args[2],i...)


################################################################################
# function Arrays.getindex!(C,A::ParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
#   iblock = first(i)
#   diagonal_getindex!(Val(all(i.==iblock)),C,A,iblock)
# end

# function diagonal_getindex!(
#   ::Val{true},
#   cache,
#   A::ArrayOfArrays{T,N},
#   iblock::Integer) where {T,N}

#   view(A.data,ArraysOfArrays._ncolons(Val(N))...,iblock)
# end

mats = [rand(3,3),rand(3,3)]
A = ParamArray(mats)
B = mats[1]

a = lazy_map(identity,A)
sum(a)

cache = array_cache(a)
r = zero(testitem(a))
for i in eachindex(a)
  println(i)
  ai = getindex!(cache,a,i)
  r += ai
end

i = 2
_cache, index_and_item = cache
index = LinearIndices(a)[i]
cg, cgi, cf = _cache
gi = getindex!(cg, a.maps, i)
index_and_item.item = Arrays._getindex_and_call!(cgi,gi,cf,a.args,i)
