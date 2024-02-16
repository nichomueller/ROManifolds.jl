using PartitionedArrays
using Test
using LinearAlgebra
using SparseArrays

ranks = with_debug() do distribute
  distribute(LinearIndices((prod(4),)))
end

n = 10
row_partition = uniform_partition(ranks,n)
col_partition = row_partition

values = map(row_partition,col_partition) do rows, cols
  i = collect(1:length(rows))
  j = i
  v = fill(2.0,length(i))
  a = sparse(i,j,v,length(rows),length(cols))
  a
end

A = PSparseMatrix(values,row_partition,col_partition)
x = pfill(3.0,col_partition)
b = similar(x,axes(A,1))
mul!(b,A,x)
map(own_values(b)) do values
  @test all( values .== 6 )
end

values_identity = map(row_partition,col_partition) do rows, cols
  i = collect(1:length(rows))
  j = i
  v = fill(1.0,length(i))
  a = sparse(i,j,v,length(rows),length(cols))
  a
end

Id = PSparseMatrix(values_identity,row_partition,col_partition)

mul_vals = map(own_values(A),own_values(Id)) do A,Id
  A * Id
end


################################################################################
using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
using Mabla.FEM
using Mabla.RB
using Mabla.Distributed
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using GridapDistributed
using PartitionedArrays
using DrWatson

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)
μ = FEM._get_params(r)[3]

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(4),)))
end
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1] + μ[2]*sin(2*π*t/μ[3])
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = sin(π*t/μ[3])
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = 0.0
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

res(μ,t,u,v,dΩ) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir("distr_toy_heateq")
info = RBInfo(dir;nsnaps_state=10,nsnaps_mdeim=5,nsnaps_test=5,save_structures=false)

sol = solve(fesolver,feop,uh0μ,r)
x = collect(sol.odesol)

μ = FEM._get_params(r)[3]

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)

_f(x,t) = f(x,μ,t)
_f(t) = x->_f(x,t)

_g(x,t) = g(x,μ,t)
_g(t) = x->_g(x,t)

_b(t,v) = ∫(_f(t)*v)dΩ
_a(t,du,v) = ∫(_a(t)*∇(v)⋅∇(du))dΩ
_m(t,dut,v) = ∫(v*dut)dΩ

_trial = TransientTrialFESpace(test,_g)
_feop = TransientAffineFEOperator(_m,_a,_b,_trial,test)
_u0 = interpolate_everywhere(x->1.0,_trial(0.0))

function Base.collect(sol::ODETools.GenericODESolution)
  ntimes = 10
  initial_values = sol.u0
  V = typeof(initial_values)
  free_values = Vector{V}(undef,ntimes)
  for (k,(ut,rt)) in enumerate(sol)
    free_values[k] = copy(ut)
  end
  return free_values
end

_sol = solve(fesolver,_feop,_u0,t0,tf)
_x = collect(_sol.odesol)

for (xi,_xi) in zip(x,_x)
  map(local_values(xi),local_values(_xi)) do xi,_xi
    @check xi[end] ≈ _xi
  end
end


######
using Gridap
using Gridap.FESpaces
using LinearAlgebra
using Test
using Mabla.FEM
using Mabla.RB
using Mabla.Distributed
using Gridap.Helpers
using Gridap.MultiField
using GridapDistributed
using PartitionedArrays

domain = (0,1,0,1)
mesh_partition = (2,2)
mesh_cells = (4,4)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(4),)))
end
model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(u,v) = ∫(∇(u)⋅∇(v))dΩ
b(v) = ∫(v)dΩ

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
trial = TrialFESpace(test,x->1)
feop = AffineFEOperator(a,b,trial,test)
solver = LUSolver()
xh = solve(solver,feop)

x = get_free_dof_values(xh)
map(local_views(x)) do x
  size(x)
end

M = assemble_matrix(a,trial,test)
map(local_views(M)) do x
  size(x)
end

solve(solver,feop)
nls = LinearFESolver(solver)
# solve(nls,op)
uh = zero(trial)
# vh,cache = solve!(uh,nls,feop)
# solve!(uh,nls,feop,nothing)

x = get_free_dof_values(uh)
op = get_algebraic_operator(feop)
# cache = solve!(x,nls.ls,op)
A = op.matrix
bb = op.vector
ss = symbolic_setup(nls.ls,A)
ns = numerical_setup(ss,A)
solve!(x,ns,bb)

a_in_main = PartitionedArrays.to_trivial_partition(A)
c = PVector{Vector{T}}(undef,partition(axes(A,2)))
c_in_main = PartitionedArrays.to_trivial_partition(c,partition(axes(a_in_main,2)))

y = get_free_dof_values(uh)
y_in_main = PartitionedArrays.to_trivial_partition(y,partition(axes(a_in_main,2)))
