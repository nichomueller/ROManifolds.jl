# module ParamFieldTests

using Gridap.Fields
using Gridap.TensorValues
using Gridap.Arrays
using ReducedOrderModels.FEM
using ReducedOrderModels.TProduct
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamFESpaces
using ReducedOrderModels.ParamSteady
using ReducedOrderModels.ParamODEs
using Test

# Parametric information

μ = Realization([[1.0],[2.0],[3.0]])
fun(x,μ) = sum(μ)
fun(μ) = x -> fun(x,μ)
funμ = 𝑓ₚ(fun,μ)


# Testing the default interface at a single point

p = Point(1.0,2.0)

f = GenericField(funμ)
df = ∇(f)
ddf = ∇∇(f)
@test typeof(f) <: GenericParamField
@test typeof(df) <: ParamFieldGradient{1}
@test typeof(ddf) <: ParamFieldGradient{2}
fp = f(p)
dfp = df(p)
ddfp = ddf(p)

# correct fields
map(f,df,ddf,μ) do f,df,ddf,μ
  𝑓 = GenericField(fun(μ))
  𝑑𝑓 = ∇(𝑓)
  𝑑𝑑𝑓 = ∇∇(𝑓)
  @test f(p) == 𝑓(p)
  @test df(p) == 𝑑𝑓(p)
  @test ddf(p) == 𝑑𝑑𝑓(p)
end

∇fp = zero(VectorValue{2,Float64})
∇fpμ = [∇fp,∇fp,∇fp]
∇∇fp = zero(TensorValue{2,2,Float64})
∇∇fpμ = [∇∇fp,∇∇fp,∇∇fp]
test_field(f,p,fp)
test_field(f,p,fp,grad=∇fpμ)
test_field(f,p,fp,grad=∇fpμ,gradgrad=∇∇fpμ)

z = evaluate(f,p)
@test z == fp

# Testing the default interface at a vector of points

np = 4
x = fill(p,np)
z = fill(p,0)

test_field(f,x,f(x))
test_field(f,x,f(x),grad=∇(f)(x))
test_field(f,x,f(x),grad=∇(f)(x),gradgrad=∇∇(f)(x))

test_field(f,z,f(z))
test_field(f,z,f(z),grad=∇(f)(z))
test_field(f,z,f(z),grad=∇(f)(z),gradgrad=∇∇(f)(z))

# integration

fun(x,μ) = 3*x[1]*sum(μ)
fun(μ) = x -> fun(x,μ)
funμ = 𝑓ₚ(fun,μ)
f = GenericField(funμ)
ϕ(x) = 2*x[1]
Φ = GenericField(ϕ)

w = ones(size(x))

i = integrate(f,x,w)
@test i == map(y->sum(y.*w),f(x))

i = integrate(f,x,w,∇(Φ))
@test i == map(y->sum(y.*w.*meas.(∇(Φ).(x))),f(x))

# Test field as collection

@test length(f) == 3
@test size(f) == (3,)
@test eltype(f) <: GenericField

# GenericField (function)

q(x,μ) = 3*x[1]*sum(μ)
q(μ) = x -> q(x,μ)
qμ = 𝑓ₚ(q,μ)

f = GenericField(qμ)
df = ∇(f)
ddf = ∇∇(f)
map(f,df,ddf,μ) do f,df,ddf,μ
  test_field(f,p,q(μ)(p))
  test_field(f,p,q(μ)(p),grad=∇(q(μ))(p))
  test_field(f,p,q(μ)(p),grad=∇(q(μ))(p),gradgrad=∇∇(q(μ))(p))

  test_field(f,p,q(μ).(p))
  test_field(f,p,q(μ).(p),grad=∇(q(μ)).(p))
  test_field(f,p,q(μ).(p),grad=∇(q(μ)).(p),gradgrad=∇∇(q(μ)).(p))

  test_field(f,z,f.(z))
  test_field(f,z,f.(z),grad=∇(f).(z))
  test_field(f,z,f.(z),grad=∇(f).(z),gradgrad=∇∇(f).(z))
end

# Operations

afun(x,μ) = sum(μ)*x[1]+2
afun(μ) = x -> afun(x,μ)
aμ = 𝑓ₚ(afun,μ)
bfun(x,μ) = sum(μ)*sin(x[1])*cos(x[2])
bfun(μ) = x -> bfun(x,μ)
bμ = 𝑓ₚ(bfun,μ)

a = GenericField(aμ)
b = GenericField(bμ)

fi = testitem(df)
li = return_cache(fi,x)
fix = evaluate!(li,fi,x)

f = Operation(*)(a,b)
@test isa(f,OperationParamField)
cp = aμ(p) .* bμ(p)
∇cp = ∇(aμ)(p) .* bμ(p) .+ aμ(p) .* ∇(bμ)(p)
test_field(f,p,cp)
test_field(f,p,cp,grad=∇cp)
test_field(f,x,f.(x))
test_field(f,x,f.(x),grad=∇(f).(x))
test_field(f,z,f.(z))
test_field(f,z,f.(z),grad=∇(f).(z))

f = Operation(/)(a,b)
cp = aμ(p) ./ bμ(p)
test_field(f,p,cp)

afun(x,μ) = (x.+2)*sum(μ)
afun(μ) = x -> afun(x,μ)
aμ = 𝑓ₚ(afun,μ)
bfun(x,μ) = 2*x*sum(μ)
bfun(μ) = x -> bfun(x,μ)
bμ = 𝑓ₚ(bfun,μ)

a = GenericField(aμ)
b = GenericField(bμ)

f = Operation(⋅)(a,b)
cp = aμ(p) .⋅ bμ(p)
∇cp = ∇(aμ)(p) .⋅ bμ(p) .+ ∇(bμ)(p) .⋅ aμ(p)
test_field(f,p,cp)
test_field(f,p,cp,grad=∇cp)
test_field(f,x,f.(x))
test_field(f,x,f.(x),grad=∇(f).(x))
test_field(f,z,f.(z))
test_field(f,z,f.(z),grad=∇(f).(z))

f = Operation(+)(a,b)
cp = aμ(p).+bμ(p)
∇cp = ∇(aμ)(p) .+ ∇(bμ)(p)
test_field(f,p,cp)
test_field(f,p,cp,grad=∇cp)
test_field(f,x,f.(x))
test_field(f,x,f.(x),grad=∇(f).(x))
test_field(f,z,f.(z))
test_field(f,z,f.(z),grad=∇(f).(z))

# Composition

mfun(g) = 3*g[1]
gfun(x,μ) = 2*x*sum(μ)
gfun(μ) = x -> gfun(x,μ)
gμ = 𝑓ₚ(gfun,μ)
ffun(x) = mfun(gμ(x))

m = GenericField(mfun)
g = GenericField(gμ)

f = m∘g
@test isa(f,OperationParamField)
fp = m(g(p))
∇fp = ∇(g)(p).⋅∇(m)(g(p))
test_field(f,p,fp)
test_field(f,p,fp,grad=∇fp)
test_field(f,x,f.(x))
test_field(f,x,f.(x),grad=∇(f).(x))
test_field(f,z,f.(z))
test_field(f,z,f.(z),grad=∇(f).(z))

vfun(x,μ) = (2*x[1]+x[2])*sum(μ)
vfun(μ) = x -> vfun(x,μ)
vμ = 𝑓ₚ(vfun,μ)

vfun1(x) = 2*x[1]+x[2]
v1 = GenericField(vfun1)
vt1 = VoidFieldMap(true)(v1)
vf1 = VoidFieldMap(false)(v1)
test_field(vt1,p,zero(v1(p)))

v = GenericField(vμ)
vt = VoidFieldMap(true)(v)
vf = VoidFieldMap(false)(v)
test_field(vt,p,zero(v(p)))
test_field(vf,p,v(p))
test_field(vt,p,zero(v(p)),grad=zero(∇(v)(p)))
test_field(vf,p,v(p),grad=∇(v)(p))
test_field(vt,p,zero(v(p)),grad=zero(∇(v)(p)),gradgrad=zero(∇∇(v)(p)))
test_field(vf,p,v(p),grad=∇(v)(p),gradgrad=∇∇(v)(p))
test_field(vt,x,zero(v(x)))
test_field(vf,x,v(x))
test_field(vt,x,zero(v(x)),grad=zero(∇(v)(x)))
test_field(vf,x,v(x),grad=∇(v)(x))
test_field(vt,x,zero(v(x)),grad=zero(∇(v)(x)),gradgrad=zero(∇∇(v)(x)))
test_field(vf,x,v(x),grad=∇(v)(x),gradgrad=∇∇(v)(x))
