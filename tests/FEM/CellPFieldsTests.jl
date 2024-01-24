# module CellPFieldsTests

using Test
using FillArrays
using Gridap.Arrays
using Gridap.TensorValues
using Gridap.Fields
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.CellData
using Gridap.FESpaces
using Random
using Mabla.FEM

domain = (0,1,0,1)
cells = (3,3)
model = CartesianDiscreteModel(domain,cells)

trian = Triangulation(model)
trian_N =BoundaryTriangulation(model)
trian_D =BoundaryTriangulation(model,tags="tag_8")
trian_S =SkeletonTriangulation(model)
trian_0 =Triangulation(model,Int32[])

μ = PRealization([[1.0],[2.0],[3.0]])
x = get_cell_points(trian)

fun(x,μ) = sum(μ)*2*x[1]
fun(μ) = x -> fun(x,μ)
funμ = 𝑓ₚ(fun,μ)

f = CellField(funμ,trian)
fx = f(x)

for (i,μ) in enumerate(μ)
  gfun(x) = 2*x[1]*sum(μ)
  g = CellField(gfun,trian)
  gx = g(x)
  @test getindex.(fx,i) == gx
end

x_0 = get_cell_points(trian_0)
fx_0 = f(x_0)
test_array(fx_0,collect(fx_0))

n_S = get_normal_vector(trian_S)
x_S = get_cell_points(trian_S)

ts = get_triangulation(∇(f))
tt = get_triangulation(n_S.plus)

@test is_change_possible(ts,tt) == true

nf_S = n_S⋅∇(f)

jnf_S = jump(n_S⋅∇(f))
jnfx_S = jnf_S(x_S)
test_array(jnfx_S,0*collect(jnfx_S))

h = CellField(rand(num_cells(trian_S)),trian_S)*jump(∇(f))
hx_S = h(x_S)
test_array(hx_S,collect(hx_S))

h = 3*mean(f)⋅jump(n_S⋅∇(f))
hx_S = h(x_S)
test_array(hx_S,0*collect(hx_S))

aa(f) = 4*f
aafx = (aa∘f)(x)
f1 = f
f2 = 2*f
b(f1,f2) = f1+f2
bbfx = (b∘(f1,f2))(x)

for (i,μ) in enumerate(μ)
  gfun(x) = 2*x[1]*sum(μ)
  g = CellField(gfun,trian)
  aagx = (aa∘g)(x)
  g1 = g
  g2 = 2*g
  bbgx = (b∘(g1,g2))(x)
  @test getindex.(aafx,i) == aagx
  @test getindex.(bbfx,i) == bbgx
end

∇fx = ∇(f)(x)
test_array(∇fx,collect(∇fx))

k = VectorValue(1.0,2.0)
∇kfx = ((∇+k)(f))(x)
test_array(∇kfx,collect(∇kfx))

α = CellField(x->2*x,trian)
βfun(x,μ) = sum(μ)*2*x[1]
βfun(μ) = x -> βfun(x,μ)
βμ = 𝑓ₚ(βfun,μ)
β1(x) = 2*x[1]
first.((βμ*α)(x)) == (β1*α)(x)
ax = ((∇+k)(βμ*α))(x)
test_array(ax,collect(ax))

# work in progress
β = CellField(βμ,trian)
ax =((∇-k)⋅β)(x)
test_array(ax,collect(ax))

ax =((∇-k)×β)(x)
test_array(ax,collect(ax))

ax =((∇-k)⊗β)(x)
test_array(ax,collect(ax))

ax =(∇.*β)(x)
test_array(ax,collect(ax))

ax =(β.*β)(x)
test_array(ax,collect(ax))

ax =((∇-k).*β)(x)
test_array(ax,collect(ax))

ax =(β⊗(∇-k))(x)
test_array(ax,collect(ax))

σ(x) = diagonal_tensor(VectorValue(1*x[1],2*x[2]))
Fields.gradient(::typeof(σ)) = x-> ThirdOrderTensorValue{2,2,2,Float64}(1,0,0,0,0,0,0,2)
ax = ((∇+k)(σ⋅α))(x)
test_array(ax,collect(ax))

h = Operation(*)(2,f)
hx = h(x)
test_array(hx,2*fx)

a = fill(2,num_cells(trian))
h = Operation(*)(a,f)
hx = h(x)
test_array(hx,2*fx)

fx = evaluate(ffun,x)
test_array(fx,r,≈)

f_N = CellField(ffun,trian_N)
x_N = get_cell_points(trian_N)
fx_N = f_N(x_N)
test_array(fx_N,collect(fx_N))

n_N = get_normal_vector(trian_N)
nx_N = n_N(x_N)
test_array(nx_N,collect(nx_N))

h = f*n_N
hx = h(x_N)
test_array(hx,collect(hx))

gfun(x) = 3*x
g = CellField(gfun,trian)

h = Operation(*)(f,g)
gx = g(x)
hx = h(x)
r = map((i,j)->broadcast(*,i,j),fx,gx)
test_array(hx,r)

h_N = Operation(*)(f_N,g)
gx_N = g(x_N)
r = map((i,j)->broadcast(*,i,j),fx_N,gx_N)
hx_N = h_N(x_N)
test_array(hx_N,r)

g_D = CellField(gfun,trian_D)

cell_h = rand(num_cells(trian))
h = CellField(cell_h,trian)
test_array(h(x),collect(h(x)))
test_array(h(x_N),collect(h(x_N)))

h_N = (2*f_N+g)⋅g
hx_N = h_N(x_N)
test_array(hx_N,collect(hx_N))

# end # module
