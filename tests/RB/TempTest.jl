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

μ = [rand(3),rand(3)]
γ(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(1/μ[3]))
γ(μ) = x->γ(x,μ)
γμ(μ) = 𝑓ₚ(γ,μ)
γμh = interpolate_everywhere(γμ(μ),trial(μ,t0))

μ1 = μ[1]
η(x) = γ(x,μ1)
ηh = interpolate_everywhere(η,trial(μ1,t0))


μ = [rand(3),rand(3)]
fs = trial(μ,t0)
object = u0μ(μ) # g(μ,t0) #
free_values = zero_free_values(fs)
dirichlet_values = zero_dirichlet_values(fs)
cell_vals = FESpaces._cell_vals(fs,object)
cell_dofs = get_cell_dof_ids(fs)
cache_vals = array_cache(cell_vals)
cache_dofs = array_cache(cell_dofs)
cells = 1:length(cell_vals)
vals = getindex!(cache_vals,cell_vals,1)
dofs = getindex!(cache_dofs,cell_dofs,1)
for (i,dof) in enumerate(dofs)
  for k in eachindex(vals)
    val = vals[k][i]
    if dof > 0
      free_values[dof] = val
    elseif dof < 0
      dirichlet_vals[-dof] = val
    else
      error("dof ids either positive or negative, not zero")
    end
  end
end

kk = g(rand(3),dt)
fe = trial(rand(3),dt)
interpolate_everywhere(kk,test)
interpolate_everywhere(u0μ(rand(3)),fe)

_cv = FESpaces._cell_vals(fe,u0μ(rand(3)))

r = realization(tpspace,nparams=10)
r1 = realization(tpspace,nparams=10,time_locations=1)
FEM.change_time!(r1,dt)

uht = solve(fesolver,feop,uh0μ)

for (u,t) in uht
  println(typeof(u))
end

function FESpaces._free_and_dirichlet_values_fill!(
  free_vals::PArray,
  dirichlet_vals::PArray,
  cache_vals,
  cache_dofs,
  cell_vals,
  cell_dofs,
  cells)

  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    map(vals,free_vals,dirichlet_vals) do vals,free_vals,dirichlet_vals
      for (i,dof) in enumerate(dofs)
        val = vals[i]
        if dof > 0
          free_vals[dof] = val
        elseif dof < 0
          dirichlet_vals[-dof] = val
        else
          error()
        end
      end
    end
  end

end

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


𝒯 = CartesianDiscreteModel((0,1,0,1),(20,20))
Ω = Interior(𝒯)
dΩ = Measure(Ω,2)
refFE = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(𝒯,refFE,dirichlet_tags="boundary")
g(x,t::Real) = 0.0
g(t::Real) = x -> g(x,t)
U = TransientTrialFESpace(V,g)
κ(t) = 1.0 + 0.95*sin(2π*t)
f(t) = sin(π*t)
res(t,u,v) = ∫( ∂t(u)*v + κ(t)*(∇(u)⋅∇(v)) - f(t)*v )dΩ
jac(t,u,du,v) = ∫( κ(t)*(∇(du)⋅∇(v)) )dΩ
jac_t(t,u,duₜ,v) = ∫( duₜ*v )dΩ
op = TransientFEOperator(res,jac,jac_t,U,V)
m(t,u,v) = ∫( u*v )dΩ
a(t,u,v) = ∫( κ(t)*(∇(u)⋅∇(v)) )dΩ
b(t,v) = ∫( f(t)*v )dΩ
op_Af = TransientAffineFEOperator(m,a,b,U,V)
linear_solver = LUSolver()
Δt = 0.05
θ = 0.5
ode_solver = ThetaMethod(linear_solver,Δt,θ)
u₀ = interpolate_everywhere(0.0,U(0.0))
t₀ = 0.0
T = 10.0
uₕₜ = solve(ode_solver,op,u₀,t₀,T)
