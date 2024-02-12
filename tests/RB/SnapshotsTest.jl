using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)

########################## HEAT EQUATION ############################

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

b(μ,t,v) = ∫(fμt(μ,t)*v)dΩ
a(μ,t,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
m(μ,t,dut,v) = ∫(v*dut)dΩ

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(m,a,b,ptspace,trial,test)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir("toy_heateq")
info = RBInfo(dir;nsnaps_state=5,nsnaps_test=0)

snaps,comp = RB.collect_solutions(info,fesolver,feop,uh0μ)

values = snaps.values
vecs = map(eachindex(values)) do i
  hcat(values[i].array...)
end
M = hcat(vecs...)
@assert M ≈ snaps

U1 = tpod(snaps)
U2 = tpod(M)
@assert U1 ≈ U2
s̃ = RB.recast_compress(U1,snaps)
@assert norm(s̃ - snaps) / norm(snaps) < 10*RB.get_tol(info)

RB._plot(trial,s̃,dir=dir)
