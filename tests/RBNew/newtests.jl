using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 0.5
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

########################## HEAT EQUATION ############################

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

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

res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
rbinfo = RBInfo(dir;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=5)

rbsolver = RBSolver(rbinfo,fesolver)

snaps,comp = RB.collect_solutions(rbsolver,feop,uh0μ)
red_trial,red_test = reduced_fe_space(rbinfo,feop,snaps)

odeop = get_algebraic_operator(feop)
pop = GalerkinProjectionOperator(odeop,red_trial,red_test)
red_lhs,red_rhs = RB.reduced_matrix_vector_form(rbsolver,pop,snaps)
red_op = reduced_operator(pop,red_lhs,red_rhs)

snaps_on = RB.select_snapshots(snaps,9:10,:)
r_on = snaps_on.realization

solve(rbsolver,red_op,r_on)

θ == 0.0 ? dtθ = dt : dtθ = dt*θ
rb_trial = get_trial(red_op)(r_on)
fe_trial = RB.get_fe_trial(red_op)(r_on)
red_x = zero_free_values(rb_trial)
y = zero_free_values(fe_trial)
z = similar(y)
z .= 0.0

ode_cache = allocate_cache(red_op,r_on)
mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(red_op,r_on,y,ode_cache)

ode_cache = update_cache!(ode_cache,red_op,r_on)

# A,b = ODETools._matrix_and_vector!(mat_cache,vec_cache,red_op,r_on,dtθ,y,ode_cache,y)
fe_A,coeff_cache,lincomb_cache = mat_cache
LinearAlgebra.fillstored!(fe_A,zero(eltype(fe_A)))
ids_all_time = RB._union_reduced_times(red_op)
A = RB.fe_matrix!(fe_A,red_op.pop,r_on,(y,y),(1,1),ode_cache)
E = RB.get_values(A[1])[1]
LHS = red_op.lhs[1][1]
RB._select_snapshots_at_space_time_locations(E,LHS,ids_all_time)
snew = RB.InnerTimeOuterParamTransientSnapshots(E)
ids_space = RB.get_indices_space(LHS)
ids_time = RB.get_indices_time(LHS)
corresponding_ids_time = filter(!isnothing,indexin(ids_all_time,ids_time))
cols = RB.col_index(snew,corresponding_ids_time,1:num_params(E))
map(x->(corresponding_ids_time.-1)*2 .+ x,1:num_params(E))
(corresponding_ids_time.-1)*2 .+ param_index
(1:num_params(E).-1)*num_times(snew) .+ corresponding_ids_time

afop = AffineOperator(A,b)
solve!(x,fesolver.nls,afop)

xrb = solve(rbsolver,feop,uh0μ)

snaps,comp = RB.collect_solutions(rbsolver,feop,uh0μ)
rbop = RB.reduced_operator(rbsolver,feop,snaps)
xrb = solve(rbsolver,rbop,snaps)

x = RB.select_snapshots(snaps,RB.online_params(rbinfo))

x - xrb
