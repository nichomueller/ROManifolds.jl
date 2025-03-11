using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.CellData
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ODEs
using ROManifolds
using ROManifolds.ParamDataStructures
using ROManifolds.ParamSteady
using Test
using FillArrays
import Gridap.MultiField: BlockMultiFieldStyle

# domain = (0,1,0,1)
# partition = (5,5)
# model = CartesianDiscreteModel(domain,partition)

# φ(x) = VectorValue(x[2],3*x[1])
# φt(x) = VectorValue(3*x[2],x[1])
# mmodel = MappedDiscreteModel(model,φ)

# Ω = Triangulation(model)
# Γ = BoundaryTriangulation(model,tags=8)
# Ωm = Triangulation(mmodel)
# Γm = BoundaryTriangulation(mmodel,tags=8)

# dΩ = Measure(Ω,4)
# dΓ = Measure(Γ,4)
# dΩm = Measure(Ωm,4)
# dΓm = Measure(Γm,4)

# g(x) = x[1]+x[2]

# reffe = ReferenceFE(lagrangian,Float64,2)
# V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
# U = TrialFESpace(V,g)
# Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
# Um = TrialFESpace(Vm,g)

# ν(x) = exp(-x[1])
# f(x) = x[2]

# atrian(u,v,dΩ) = ∫(ν*∇(v)⋅∇(u))dΩ
# btrian(v,dΩ,dΓ) = ∫(f*v)dΩ + ∫(f*v)dΓ

# a(u,v) = atrian(u,v,dΩ)
# b(v) = btrian(v,dΩ,dΓ)
# am(u,v) = atrian(u,v,dΩm)
# bm(v) = btrian(v,dΩm,dΓm)

# op = AffineFEOperator(a,b,U,V)
# opm = AffineFEOperator(am,bm,Um,Vm)

# uh = solve(op)
# uhm = solve(opm)

# v = get_fe_basis(V)
# u = get_trial_fe_basis(V)
# jcell = a(u,v)[Ω]

# vm = get_fe_basis(Vm)
# um = get_trial_fe_basis(Vm)
# jcellm = am(um,vm)[Ωm]

# detJφ = 3
# Jφt = CellField(∇(φ),Ω)
# νm = ν∘φ
# mappedj = (∫( νm*(inv(Jφt)⋅∇(v)) ⋅ (inv(Jφt)⋅∇(u))*detJφ )dΩ)[Ω]

# ncells = num_cells(Ω)
# compare = lazy_map(≈,jcellm,mappedj)
# @assert sum(compare) == ncells


#

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

pspace = ParamSpace((3.0,4.0))

μ = Realization([[3.0],[4.0]])
ϕ(μ) = x->VectorValue(x[2],μ[1]*x[1])
ϕμ(μ) = parameterize(ϕ,μ)
mmodel = MappedDiscreteModel(model,ϕμ(μ))

Ωm = Triangulation(mmodel)
Γm = BoundaryTriangulation(mmodel,tags=8)

dΩm = Measure(Ωm,4)
dΓm = Measure(Γm,4)

g(μ) = x->x[1]+μ[1]*x[2]
gμ(μ) = parameterize(g,μ)

reffe = ReferenceFE(lagrangian,Float64,2)
Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
Um = ParamTrialFESpace(Vm,gμ)

Umμ = Um(μ)

ν(μ) = x->x[1]+μ[1]*x[2]
νμ(μ) = parameterize(ν,μ)
f(μ) = x->x[1]+μ[1]*x[2]
fμ(μ) = parameterize(f,μ)

am(μ,u,v) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩm
bm(μ,u,v) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩm - ∫(fμ(μ)*v)dΩm #+ ∫(fμ(μ)*v)dΓm

opm = LinearParamOperator(bm,am,pspace,Um,Vm)
# uhm = solve(opm)

xm, = solve(LUSolver(),opm,μ)

μ1 = 3.0
mmodel1 = MappedDiscreteModel(model,ϕ(μ1))
Ωm1 = Triangulation(mmodel1)
dΩm1 = Measure(Ωm1,4)
Vm1 = TestFESpace(mmodel1,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
Um1 = TrialFESpace(Vm1,g(μ1))
am1(u,v) = ∫(ν(μ1)*∇(v)⋅∇(u))dΩm1
bm1(v) = ∫(f(μ1)*v)dΩm1
opm1 = AffineFEOperator(am1,bm1,Um1,Vm1)
xm1 = solve(LUSolver(),opm1)

@assert xm[1] ≈ get_free_dof_values(xm1)

# v = get_fe_basis(Vm)
# u = get_trial_fe_basis(Vm)
# νμ(μ)*∇(v)⋅∇(u)

# cell_∇a = lazy_map(Broadcasting(∇),get_data(v))
# cell_map = get_cell_map(get_triangulation(v))
# # lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
# cell_Jt = lazy_map(∇,cell_map)
# cell_invJt = lazy_map(Operation(pinvJt),cell_Jt)
# k = Broadcasting(Operation(⋅))
# args = cell_invJt,cell_∇a
# fi = map(testitem,args)
# T = return_type(k,fi...)

# ∇(v)⋅∇(u)

# trian = Ωm

# degree = 2
# quad = CellQuadrature(trian,degree)

# x = get_cell_points(quad)
# @test isa(x,CellPoint)

# v = GenericCellField(get_cell_shapefuns(trian),trian,ReferenceDomain())
# u = GenericCellField(lazy_map(transpose,get_data(v)),v.trian,v.domain_style)

# m = ∇(v)⋅∇(u)
# s = integrate(m,quad)
# s = ∫( ∇(v)⋅∇(u) )*quad
# s = ∫(1)*quad

# trian_N = BoundaryTriangulation(mmodel)
# quad_N = CellQuadrature(trian_N,degree)

# x_N = get_cell_points(quad_N)

# s = ∫( ∇(v)⋅∇(u) )*quad_N

# s = ∫(1)*quad_N
# s = ∫( x->1 )*quad_N

# quad_N = CellQuadrature(trian_N,degree,T=Float32)
# @test eltype(quad_N.cell_point) == Vector{Point{num_dims(trian_N),Float32}}
# @test eltype(quad_N.cell_weight) == Vector{Float32}

# s = ∫( ∇(v)⋅∇(u) )*quad_N
# test_array(s,collect(s))

# s = ∫(1)*quad_N
# @test sum(s) ≈ 6
# @test ∑(s) ≈ 6

# s = ∫( x->1 )*quad_N
# @test sum(s) ≈ 6
# @test ∑(s) ≈ 6

# cell_measure = get_cell_measure(trian)
# cell_measure_N = get_cell_measure(trian_N,trian)
# @test length(cell_measure) == num_cells(model)
# @test length(cell_measure_N) == num_cells(model)
# @test sum(cell_measure) ≈ 1
# @test sum(cell_measure_N) ≈ 6

# quad = CellQuadrature(trian,Quadrature(duffy,2))
# s = ∫(1)*quad
# @test sum(s) ≈ 1
# @test ∑(s) ≈ 1

# quad = CellQuadrature(trian,Quadrature(TET,duffy,2))
# s = ∫(1)*quad
# @test sum(s) ≈ 1
# @test ∑(s) ≈ 1
