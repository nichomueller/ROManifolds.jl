using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using Mabla
using Mabla.FEM

root = pwd()
model = DiscreteModelFromFile(joinpath(root,"models/elasticity_3cyl2D.json"))
test_path = "$root/results/HeatEquation/elasticity_3cyl2D"
order = 1
degree = 2*order
Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = 𝑓ₚₜ(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = 𝑓ₚₜ(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = 𝑓ₚₜ(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)

u0(x,μ) = 0.0
u0(μ) = x->u0(x,μ)
u0μ(μ) = 𝑓ₚ(u0,μ)

res(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v) = ∫(v*dut)dΩ

pranges = fill([1.,10.],3)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
tdomain = t0:dt:tf
tpspace = TransientParametricSpace(pranges,tdomain)

T = Float
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialPFESpace(test,g)
feop = AffineTransientPFEOperator(res,jac,jac_t,tpspace,trial,test)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),θ,dt)

solve(fesolver,feop,uh0μ)

r = realization(feop.tpspace;nparams=1)
params = FEM.get_parameters(r)
ode_op = get_algebraic_operator(feop)
uu0 = get_free_dof_values(uh0μ(params))
ode_sol = solve(solver,ode_op,uu0,r)

ϵ = 1e-4
load_solutions = false
save_solutions = true
load_structures = false
save_structures = true
postprocess = true
norm_style = :l2
nsnaps_state = 50
nsnaps_mdeim = 20
nsnaps_test = 10
st_mdeim = false
rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
rbspace = reduced_basis(rbinfo,feop,sols)



abstract type ReducedFESpace <: FESpace end
struct ReducedSingleFieldFESpace{F,R} <: ReducedFESpace
  fe::F
  reduced_basis::R
end



w = (u*v)
cache = return_cache(w,x)
@which evaluate!(cache,w,x)
u(x)

boh = ∫(a(rand(3),dt)*∇(φ)⋅∇(φ))dΩ
boh[Ω]

φᵢ = FEFunction(test,bs1)
φⱼ = FEFunction(test,bs1)
@time for bsi in eachcol(bs)
  for bsj in eachcol(bs)
    ∫(a(rand(3),dt)*∇(φᵢ)⋅∇(φⱼ))dΩ
  end
end

trial0 = trial(nothing)
@time begin
  μ = rand(3)
  A = assemble_matrix((φᵢ,φⱼ)->∫(a(μ,dt)*∇(φᵢ)⋅∇(φⱼ))dΩ,trial0,test)
  bs'*A*bs
end

(φᵢ*φᵢ)(x)
fs,free_values,dirichlet_values = test,bs1,get_dirichlet_dof_values(test)
cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
cell_field = CellField(fs,cell_vals)
SingleFieldFEFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)

struct DummyFunction
end

#############################
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools

𝒯 = CartesianDiscreteModel((0,1,0,1),(20,20))
Ω = Interior(𝒯)
dΩ = Measure(Ω,2)
T = Float64
reffe_u = ReferenceFE(lagrangian,T,2)
reffe_p = ReferenceFE(lagrangian,T,1)
g(x,t::Real) = 0.0
g(t::Real) = x -> g(x,t)
mfs = BlockMultiFieldStyle()
test_u = TestFESpace(𝒯,reffe_u;conformity=:H1,dirichlet_tags="boundary")
trial_u = TransientTrialFESpace(test_u,g)
test_p = TestFESpace(𝒯,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
Yb  = TransientMultiFieldFESpace([test_u,test_p];style=mfs)
Xb  = TransientMultiFieldFESpace([trial_u,trial_p];style=mfs)
κ(t) = 1.0 + 0.95*sin(2π*t)
f(t) = sin(π*t)
res(t,(u,p),(v,q)) = ∫( ∂t(u)*v + κ(t)*(∇(v)⊙∇(u)) - p*(∇⋅(v)) - q*(∇⋅(u)) - f(t)*v )dΩ
jac(t,(u,p),(du,dp),(v,q)) = ∫( κ(t)*(∇(du)⋅∇(v)) - dp*(∇⋅(v)) - q*(∇⋅(du)) )dΩ
jac_t(t,(u,p),(duₜ,dpₜ),(v,q)) = ∫( duₜ*v )dΩ
op = TransientFEOperator(res,jac,jac_t,U,V)
m(t,u,v) = ∫( u*v )dΩ
a(t,u,v) = ∫( κ(t)*(∇(u)⋅∇(v)) )dΩ
b(t,v) = ∫( f(t)*v )dΩ
op_Af = TransientAffineFEOperator(m,a,b,U,V)
linear_solver = LUSolver()
Δt = 0.1
θ = 0.5
ode_solver = ThetaMethod(linear_solver,Δt,θ)
u₀ = interpolate_everywhere(0.0,U(0.0))
t₀ = 0.0
T = 10.0
uₕₜ = solve(ode_solver,op,u₀,t₀,T)

g0(x) = 0.0
trial_u = TrialFESpace(test_u,g0)
trial_p = TrialFESpace(test_p)
Yb  = MultiFieldFESpace([test_u,test_p];style=mfs)
Xb  = MultiFieldFESpace([trial_u,trial_p];style=mfs)
biform((u1,u2),(v1,v2)) = ∫(∇(u1)⋅∇(v1) + u2⋅v2 - u1⋅v2)*dΩ
liform((v1,v2)) = ∫(v1 - v2)*dΩ
ub = get_trial_fe_basis(Xb)
vb = get_fe_basis(Yb)
bdata = collect_cell_matrix_and_vector(Xb,Yb,biform(ub,vb),liform(vb))
bmatdata = collect_cell_matrix(Xb,Yb,biform(ub,vb))
bvecdata = collect_cell_vector(Yb,liform(vb))
