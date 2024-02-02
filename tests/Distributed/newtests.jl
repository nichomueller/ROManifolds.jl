using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM
using Mabla.Distributed
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using GridapDistributed
using PartitionedArrays

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
mesh_partition = (2,2)
mesh_cells = (4,4)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(4),)))
end
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

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
fesolver = ThetaMethod(LUSolver(),dt,θ)

sol = solve(fesolver,feop,uh0μ,r)

for (uh,rt) in sol
end

sol = sol.odesol
wf = copy(sol.u0)
w0 = copy(sol.u0)
r0 = FEM.get_at_time(sol.r,:initial)
cache = nothing
# uf,rf,cache = solve_step!(wf,sol.solver,sol.op,r0,w0,cache)
dt = sol.solver.dt
θ = sol.solver.θ
θ == 0.0 ? dtθ = dt : dtθ = dt*θ
FEM.shift_time!(r,dtθ)
ode_cache = allocate_cache(sol.op,r0)
vθ = similar(w0)
vθ .= 0.0
l_cache = nothing
A,bb = ODETools._allocate_matrix_and_vector(sol.op,r0,w0,ode_cache)
ode_cache = update_cache!(ode_cache,sol.op,r0)
# ODETools._matrix_and_vector!(A,bb,sol.op,r0,dtθ,w0,ode_cache,vθ)

# ODETools._matrix!(A,sol.op,r0,dtθ,w0,ode_cache,vθ)
z = zero(eltype(A))
LinearAlgebra.fillstored!(A,z)
# jacobians!(A,odeop,r0,(vθ,vθ),(1.0,1/dtθ),ode_cache)
Xh, = ode_cache
dxh = ()
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],vθ))
end
xh=TransientCellField(EvaluationFunction(Xh[1],vθ),dxh)
# ODETools.jacobians!(A,feop,r0,xh,(1.0,1/dtθ),ode_cache)
_matdata_jacobians = TransientFETools.fill_jacobians(feop,r0,xh,(1.0,1/dtθ))
matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
# assemble_matrix_add!(A,feop.assem,matdata)
# numeric_loop_matrix!(A,feop.assem,matdata)
assem = FEM.get_param_assembler(feop.assem,r0)
rows = get_rows(assem)
cols = get_cols(assem)
# map(numeric_loop_matrix!,local_views(A,rows,cols),local_views(assem),matdata)
B = local_views(A,rows,cols).items[1]
assem = local_views(assem).items[1]
matdata = matdata.items[1]
matdataA = (matdata[1][1],matdata[2][1],matdata[3][1])
numeric_loop_matrix!(B,assem,matdata)
