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
using Mabla.RB

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
aμt(μ,t) = PTFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = PTFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = PTFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = PFunction(u0,μ)

res(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v) = ∫(v*dut)dΩ

pranges = fill([1.,10.],3)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
tdomain = t0:dt:tf
ptspace = TransientParametricSpace(pranges,tdomain)

T = Float
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialPFESpace(test,g)
feop = AffineTransientPFEOperator(res,jac,jac_t,ptspace,trial,test)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

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

trial0 = trial(nothing,nothing)
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
