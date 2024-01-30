import Gridap
using Gridap.Helpers

struct D{T,N,A<:AbstractVector{<:AbstractArray{T,N}},L} <: AbstractArray{T,N}
  array::A
  function D(array::A,::Val{L}) where {T,N,A<:AbstractVector{<:AbstractArray{T,N}},L}
    new{T,N,A,L}(array)
  end
end

function D(array)
  D(array,Val(length(array)))
end

get_array(a::D) = a.array
Base.length(::D{T,N,A,L}) where {T,N,A,L} = L
Base.length(::Type{D{T,N,A,L}}) where {T,N,A,L} = L
Base.size(a::D) = (length(a),)
Base.axes(a::D) = (Base.OneTo(length(a)),)
Base.eltype(::D{T,N,A,L}) where {T,N,A,L} = T
Base.eltype(::Type{D{T,N,A,L}}) where {T,N,A,L} = T
Base.ndims(::D{T,N,A,L}) where {T,N,A,L} = N
Base.ndims(::Type{D{T,N,A,L}}) where {T,N,A,L} = N
Base.first(a::D) = a[1]
Base.eachindex(::D{T,N,A,L}) where {T,N,A,L} = Base.OneTo(L)
Base.lastindex(::D{T,N,A,L}) where {T,N,A,L} = L
Base.getindex(a::D,i...) = get_array(a)[i...]
Base.setindex!(a::D,v,i...) = get_array(a)[i...] = v
Base.iterate(a::D,i...) = iterate(get_array(a),i...)

function Base.show(io::IO,::MIME"text/plain",a::D{T,N,A,L}) where {T,N,A,L}
  println(io, "Parametric vector of types $(eltype(A)) and length $L, with entries:")
  show(io,a.array)
end

struct DBroadcast{D}
  array::D
end

get_array(d::DBroadcast) = d.array

# function Base.broadcasted(f,a::D)
#   Base.broadcasted(f,get_array(a))
# end

function Base.broadcasted(f,a::Union{D,DBroadcast}...)
  bc = map((x...)->Base.broadcasted(f,x...),map(get_array,a)...)
  DBroadcast(bc)
end

function Base.broadcasted(f,a::Union{D,DBroadcast},b::Number)
  bc = map(a->Base.broadcasted(f,a,b),a.array)
  DBroadcast(bc)
end

function Base.broadcasted(f,a::Number,b::Union{D,DBroadcast})
  bc = map(b->Base.broadcasted(f,a,b),b.array)
  DBroadcast(bc)
end

function Base.broadcasted(f,
  a::Union{D,DBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{D,DBroadcast})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::DBroadcast)
  a = map(Base.materialize,b.array)
  D(a)
end

function Base.materialize!(a::D,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),a.array)
  a
end

function Base.materialize!(a::D,b::DBroadcast)
  map(Base.materialize!,a.array,b.array)
  a
end

v1 = [1,3,5]
v2 = [2,4,6]
vv = [v1,v2]
d = D(vv)

# @check maximum.(d) == [5,6] # reduction not supported :(

@check d .* -1 == D([[-1,-3,-5],[-2,-4,-6]])
@check d == D([[1,3,5],[2,4,6]])

d .*= -1 # ok
@check d == D([[-1,-3,-5],[-2,-4,-6]])
@check d .* -1 == D([[1,3,5],[2,4,6]])

d .= 0 # not ok
@check d == D([[0,0,0],[0,0,0]])

w1 = [1,4,5]
w2 = [2,3,6]
ww = [w1,w2]
b = D(ww)

b .= d

@. b = 3*d - 2*d

a1 = Base.broadcasted(/, 1, 2)
b1 = Base.broadcasted(/, 1, 2)
a2 = Base.broadcasted(*, a1, d)
a3 = Base.broadcasted(*, b1, d)
a4 = Base.broadcasted(-, a2, a3)
Base.materialize!(b, a4)

bc = Base.broadcasted(Base.identity,d)
Base.materialize!(b,bc)


################
using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools

θ = 0.2
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
μt = realization(ptspace,nparams=3)
μ = get_params(μt)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

g(x,μ,t) = 1
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
reffe = ReferenceFE(lagrangian,Float64,order)
test = FESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
trial = TransientTrialParamFESpace(test,gμt)

using Gridap.Algebra: NewtonRaphsonSolver
linear = false
if linear
  b(μ,t,v) = ∫(gμt(μ,t)*v)dΩ
  a(μ,t,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
  m(μ,t,dut,v) = ∫(v*dut)dΩ
  feop = AffineTransientParamFEOperator(m,a,b,ptspace,trial,test)
  nls = LUSolver()
else
  res(μ,t,u,v) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(gμt(μ,t)*v)dΩ
  jac(μ,t,u,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
  jac_t(μ,t,u,dut,v) = ∫(v*dut)dΩ
  feop = TransientParamFEOperator(res,jac,jac_t,ptspace,trial,test)
  tol = 1e-10
  maxiters = 20
  nls = NewtonRaphsonSolver(ls,tol,maxiters)
end

ode_solver = ThetaMethod(nls,dt,θ)

ic = interpolate_everywhere(u0μ(μ),trial(μ,t0))
odeop = get_algebraic_operator(feop)
w0 = get_free_dof_values(ic)
wf = copy(w0)
w0 = copy(w0)
r0 = FEM.get_at_time(μt,:initial)
cache = nothing

sol = solve(ode_solver,feop,μ -> interpolate_everywhere(u0μ(μ),trial(μ,t0)))
for (uf,rf) in sol
  println(get_times(rf))
end

odesol = solve(ode_solver,odeop,w0,μt)
x,state=iterate(odesol,state)
for (uf,rf) in odesol
  println(get_times(rf))
end

function Gridap.Algebra._solve_nr!(x::ParamArray,A,b,dx,ns,nls,op)
  function my_check_convergence(nls,b)
    m = maximum(abs,b)
    return false,m
  end
  function my_check_convergence(nls,b,m0)
    m = maximum(abs,b)
    println(maximum(m ./ m0))
    return all(m .< nls.tol * m0)
  end

  # Check for convergence on the initial residual
  isconv, conv0 = my_check_convergence(nls,b)
  if isconv; return; end

  # Newton-like iterations
  for nliter in 1:nls.max_nliters

    # Solve linearized problem
    rmul!(b,-1)
    solve!(dx,ns,b)
    x .+= dx

    # Check convergence for the current residual
    residual!(b, op, x)
    isconv = my_check_convergence(nls, b, conv0)
    if isconv; return; end

    if nliter == nls.max_nliters
      @unreachable
    end

    # Assemble jacobian (fast in-place version)
    # and prepare solver
    jacobian!(A, op, x)
    numerical_setup!(ns,A)

  end

end

w0 = get_free_dof_values(ic)
wf = copy(w0)
w0 = copy(w0)
rf = FEM.get_at_time(r,:initial)
cache = nothing

wf,rf,cache = solve_step!(wf,ode_solver,odeop,rf,w0,cache)

w0 .= wf
state = (wf,w0,rf,cache)

get_times(rf) >= 100*eps()*FEM.get_final_time(r)

# uf,rf,cache = solve_step!(wf,ode_solver,odeop,r0,w0,cache)

θ == 0.0 ? dtθ = dt : dtθ = dt*θ
r = FEM.get_at_time(μt,:initial)

if isnothing(cache)
  ode_cache = TransientFETools.allocate_cache(odeop,r)
  vθ = similar(w0)
  vθ .= 0.0
  nl_cache = nothing
else
  ode_cache,vθ,nl_cache = cache
end

ode_cache = update_cache!(ode_cache,odeop,r)

nlop = ThetaMethodParamOperator(odeop,r,dtθ,w0,ode_cache,vθ)

nl_cache = solve!(wf,ode_solver.nls,nlop,nl_cache)

if 0.0 < θ < 1.0
  @. wf = wf*(1.0/θ)-w0*((1-θ)/θ)
end

cache = (ode_cache,vθ,nl_cache)
FEM.shift_time!(r,dt*(1-θ))


#########################
using Gridap.TensorValues
using Gridap.Arrays
using Gridap.Fields
using Gridap.Fields: ArrayBlock, MockFieldArray, MockField, BroadcastingFieldOpMap, BlockMap
using Test
using FillArrays
using LinearAlgebra
using Mabla.FEM

b = ParamArray([[1,2,3,4],[5,6,7,8]])
@test length(b) == 2
@test size(b) == (4,)
@test ndims(b) == 1
@test b[1] == [1,2,3,4]
a = ParamArray([rand(4,3) for i in 1:2])

@test typeof(testvalue(typeof(b))) == typeof(b)

x = [Point(0.5,0.5),Point(1.0,0.0)]
w = [1.0,1.0]
f_basis = MockFieldArray(rand(4))
f_array = Vector{typeof(f_basis)}(undef,3)
f_array[2] = f_basis
f_array[3] = f_basis
f_touched = [false,true,true]
f = ArrayBlock(f_array,f_touched)
u = ParamArray([rand(4) for i in 1:2])
uh = linear_combination(u,f)
fx = evaluate(f,x)
ft = transpose(f)
ftx = evaluate(ft,x)
uhx = evaluate(uh,x)
∇f = Broadcasting(∇)(f)
∇ft = Broadcasting(∇)(ft)
∇fx = evaluate(∇f,x)
∇uh = Broadcasting(∇)(uh)
∇uhx = evaluate(∇uh,x)
intf = integrate(f,x,w)
ϕ = MockField(VectorValue(2.1,3.4))
g = Broadcasting(∘)(f,ϕ)
#gx = evaluate(g,x)
h_basis = MockFieldArray([1.0+im,3.0*im,-1.0*im,1.0*im])
h_array = Vector{typeof(h_basis)}(undef,3)
h_array[1] = h_basis
h_array[3] = h_basis
h_touched = [true,false,true]
h = ArrayBlock(h_array,h_touched)
Jt = ∇(ϕ)
invJt = Operation(Fields.pinvJt)(Jt)
Broadcasting(Operation(⋅))(invJt,∇f)
Broadcasting(Operation(⋅))(invJt,∇ft)


########################## SETTING #############################

using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField

θ = 0.5
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)
μ = FEM._get_params(r)[1]

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

########################## HEAT EQUATION ############################

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

b(μ,t,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
a(μ,t,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
m(μ,t,dut,v) = ∫(v*dut)dΩ

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(m,a,b,ptspace,trial,test)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),θ,dt)

sol = solve(fesolver,feop,uh0μ,r)

# gridap

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_f(x,t) = f(x,μ,t)
_f(t) = x->_f(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_h(x,t) = h(x,μ,t)
_h(t) = x->_h(x,t)

_b(t,v) = ∫(_f(t)*v)dΩ + ∫(_h(t)*v)dΓn
_a(t,du,v) = ∫(_a(t)*∇(v)⋅∇(du))dΩ
_m(t,dut,v) = ∫(v*dut)dΩ

_trial = TransientTrialFESpace(test,_g)
_feop = TransientAffineFEOperator(_m,_a,_b,_trial,test)
_u0 = interpolate_everywhere(x->0.0,_trial(0.0))

_sol = solve(fesolver,_feop,_u0,t0,tf)

for ((uh,rt),(_uh,_t)) in zip(sol,_sol)
  uh1 = FEM._getindex(uh,1)
  t = get_times(rt)
  @check t == _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "$(get_free_dof_values(uh1)) != $(get_free_dof_values(_uh))"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end

########################## STOKES ############################

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = VectorValue(0.0,0.0)
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0.0)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

b(μ,t,(v,q)) = ∫(fμt(μ,t)⋅v)dΩ
a(μ,t,(du,dp),(v,q)) = ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
m(μ,t,(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial_u = TransientTrialParamFESpace(test_u,gμt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = AffineTransientParamFEOperator(m,a,b,ptspace,trial,test)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),θ,dt)

sol = solve(fesolver,feop,xh0μ,r)
iterate(sol)

# gridap

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_f(x,t) = f(x,μ,t)
_f(t) = x->_f(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_b(t,(v,q)) = ∫(_f(t)⋅v)dΩ
_a(t,(du,dp),(v,q)) = ∫(_a(t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
_m(t,(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ

_trial_u = TransientTrialFESpace(test_u,_g)
_trial = TransientMultiFieldFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())
_feop = TransientAffineFEOperator(_m,_a,_b,_trial,test)
_x0 = interpolate_everywhere([u0(μ),p0(μ)],_trial(0.0))

_sol = solve(fesolver,_feop,_x0,t0,tf)
iterate(_sol)

for ((xh,rt),(_xh,_t)) in zip(sol,_sol)
  uh,ph = xh
  uh1 = FEM._getindex(uh,1)
  ph1 = FEM._getindex(ph,1)
  _uh,_ph = _xh
  t = get_times(rt)
  @check t == _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "failed at time $t"
  @check get_free_dof_values(ph1) ≈ get_free_dof_values(_ph) "failed at time $t"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end

for ((xh,rt),(_xh,_t)) in zip(sol.odesol,_sol.odesol)
  t = get_times(rt)
  @check t == _t "$t != $_t"
  @check xh[1] ≈ _xh "$(xh[1]) != $_xh"
end

############################# NAVIER STOKES ###################################

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = VectorValue(0.0,0.0)
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0.0)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

form(μ,t,(u,p),(v,q)) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
res(μ,t,(u,p),(v,q)) = ∫(v⋅∂t(u))dΩ + form(μ,t,(u,p),(v,q)) + c(u,v)
jac(μ,t,(u,p),(du,dp),(v,q)) = form(μ,t,(du,dp),(v,q)) + dc(u,du,v)
jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial_u = TransientTrialParamFESpace(test_u,gμt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p])
trial = TransientMultiFieldParamFESpace([trial_u,trial_p])
feop = TransientParamFEOperator(res,jac,jac_t,ptspace,trial,test)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
nls = Gridap.Algebra.NewtonRaphsonSolver(LUSolver(),1e-10,20)
fesolver = ThetaMethod(nls,θ,dt)

sol = solve(fesolver,feop,xh0μ,r)
iterate(sol)

# gridap

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_form(t,(u,p),(v,q)) = ∫(_a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
_res(t,(u,p),(v,q)) = ∫(v⋅∂t(u))dΩ + _form(t,(u,p),(v,q)) + c(u,v)
_jac(t,(u,p),(du,dp),(v,q)) = _form(t,(du,dp),(v,q)) + dc(u,du,v)
_jac_t(t,(u,p),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ

_trial_u = TransientTrialFESpace(test_u,_g)
_trial = TransientMultiFieldFESpace([_trial_u,trial_p])
_feop = TransientFEOperator(_res,_jac,_jac_t,_trial,test)
_x0 = interpolate_everywhere([u0(μ),p0(μ)],_trial(0.0))

_sol = solve(fesolver,_feop,_x0,t0,tf)
iterate(_sol)

for ((xh,rt),(_xh,_t)) in zip(sol,_sol)
  uh,ph = xh
  uh1 = FEM._getindex(uh,1)
  ph1 = FEM._getindex(ph,1)
  _uh,_ph = _xh
  t = get_times(rt)
  @check t == _t "$t != $_t"
  @check get_free_dof_values(uh1) ≈ get_free_dof_values(_uh) "failed at time $t"
  @check get_free_dof_values(ph1) ≈ get_free_dof_values(_ph) "failed at time $t"
  @check uh1.dirichlet_values ≈ _uh.dirichlet_values
end
