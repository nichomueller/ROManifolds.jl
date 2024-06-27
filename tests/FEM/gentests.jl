v = rand(100)
pv = array_of_consecutive_zero_arrays(v,10)

nparams = 60
sol = solve(fesolver,feop,uh0μ;nparams)
odesol = sol.odesol
r = odesol.r
μ = get_params(r)
xh = uh0μ(μ)
sol = odesol

using Gridap.ODEs
r = ParamDataStructures.get_at_time(r,:initial)
cache = allocate_odecache(sol.solver,sol.odeop,r,sol.us0)
state0,cache = ode_start(sol.solver,sol.odeop,r,sol.us0,cache)
statef = copy.(state0)
rf,statef,cache = ode_march!(statef,sol.solver,sol.odeop,r,state0,cache)

w0 = state0[1]
odeslvrcache,odeopcache = cache
reuse,A,b,sysslvrcache = odeslvrcache

sysslvr = fesolver.sysslvr

x = statef[1]
fill!(x,zero(eltype(x)))
dtθ = θ*dt
shift!(r,dtθ)
usx = (w0,x)
ws = (dtθ,1)

update_odeopcache!(odeopcache,sol.odeop,r)

stageop = LinearParamStageOperator(sol.odeop,odeopcache,r,usx,ws,A,b,reuse,sysslvrcache)

# sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)
using LinearAlgebra
A = stageop.A
numerical_setup!(sysslvrcache,A)
b = stageop.b
rmul!(b,-1)
# solve!(x,sysslvrcache,b)
ldiv!(x,sysslvrcache.factors,b)

ldiv!(x.data[:,1],sysslvrcache.factors[1],b.data[:,1])
y = view(x.data,:,1)
ldiv!(y,sysslvrcache.factors[1],b.data[:,1])

x′ = array_of_similar_arrays(x.data[:,1],60)
ldiv!(x′.data[1],sysslvrcache.factors[1],b.data[:,1])

# state0,cache = ode_start(sol.solver,sol.odeop,r,sol.us0,cache)

# statef = copy.(state0)
# rf,statef,cache = ode_march!(statef,sol.solver,sol.odeop,r,state0,cache)

# uf = copy(first(sol.us0))
# uf,cache = ode_finish!(uf,sol.solver,sol.odeop,r,rf,statef,cache)
