using Test
using FillArrays
using Gridap.Helpers
using Gridap.Fields
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.CellData
using Gridap.TensorValues
using Gridap.Geometry
using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces

domain = (0,1,0,1)
cells = (2,2)
model = simplexify(CartesianDiscreteModel(domain,cells))
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

v = GenericCellField(get_cell_shapefuns(Ω),Ω,ReferenceDomain())
u = GenericCellField(lazy_map(transpose,get_data(v)),Ω,ReferenceDomain())

μ = ParamRealization([[1],[2],[3]])
μ₀ = ParamRealization([[0],[0],[0]])
f(x,μ) = 1+sum(μ)
f(μ) = x -> f(x,μ)
fμ = 𝑓ₚ(f,μ)
fμ₀ = 𝑓ₚ(f,μ₀)

aa = ∫(fμ*u*v)*dΩ + ∫(fμ*∇(u)⋅∇(v))*dΩ

cf1 = fμ*u
# cf = cf1*v
args = cf1,v
x = CellData._get_cell_points(args...)
ax = map(i->i(x),args)
axi = map(first,ax)
fi = Fields.BroadcastingFieldOpMap(*)
r = fi(axi...)

# axi1 = map(testitem,axi)
c1 = return_cache(fi,axi...)
evaluate!(c1,fi,axi...)

cell_field = get_data(cf1)
cell_point = get_data(x)
# α = cf1(x)
cα = return_cache(testitem(cell_field),testitem(cell_point))

# fμ*u
args = CellData._convert_to_cellfields(fμ,u)
x = CellData._get_cell_points(args...)
ax = map(i->i(x),args)
axi = map(first,ax)
fi = Fields.BroadcastingFieldOpMap(*)
# r = fi(axi...)
# cα = return_cache(fi,axi...)
# evaluate!(cα,fi,axi...)
A,b = axi
c = return_cache(fi,testitem(A),b)
cx = evaluate!(c,fi,testitem(A),b)
cache = Vector{typeof(c)}(undef,param_length(A))
data = Vector{typeof(cx)}(undef,param_length(A))
@inbounds for i = param_eachindex(A)
  cache[i] = return_cache(fi,param_getindex(A,i),b)
end
data = ParamArray(data)

@inbounds for i = param_eachindex(A)
  data[i] = evaluate!(cache[i],fi,param_getindex(A,i),b)
end

# args = CellData._convert_to_cellfields(fμ,∇(u))
aa = fμ,∇(u)
a1 = filter(i->isa(i,CellField),aa)
a2 = CellData._to_common_domain(a1...)
target_domain = DomainStyle(first(a2))
target_trian = get_triangulation(first(a2))
map(i->CellField(i,target_trian,target_domain),aa)

t1 =  @btime filter(i->isa(i,CellField),aa)
t2 =  @btime CellData._to_common_domain(a1...)
t3 =  @btime DomainStyle(first(a2))
t4 =  @btime get_triangulation(first(a2))
t5 =  @btime map(i->CellField(i,target_trian,target_domain),aa)

# ∇(v)
cell_∇a = lazy_map(Broadcasting(∇),get_data(v))
if DomainStyle(v) == PhysicalDomain()
  g = cell_∇a
else
  cell_map = get_cell_map(get_triangulation(v))
  gg = lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
end
similar_cell_field(v,g,get_triangulation(v),DomainStyle(v))

# lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
