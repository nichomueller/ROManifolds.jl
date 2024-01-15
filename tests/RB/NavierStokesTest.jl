module NavierStokesTest

using Gridap
using Mabla
import Gridap.Algebra: NewtonRaphsonSolver

root = pwd()
model = DiscreteModelFromFile(joinpath(root,"models/model_circle_2D_coarse.json"))
test_path = "$root/results/NavierStokes/model_circle_2D_coarse"
order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

ranges = fill([1.,10.],3)
sampling = UniformSampling()
pspace = ParametricSpace(ranges,sampling)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientPFunction(a,μ,t)

g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
g(μ,t) = x->g(x,μ,t)
g0(x,μ,t) = VectorValue(0,0)
g0(μ,t) = x->g0(x,μ,t)

u0(x,μ) = VectorValue(0,0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = PFunction(u0,μ)
p0(x,μ) = 0
p0(μ) = x->p0(x,μ)
p0μ(μ) = PFunction(p0,μ)

c(μ,t,(u,p),(v,q)) = ∫ₚ(v⊙(∇(u)'⋅u),dΩ)
dc(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(v⊙(∇(du)'⋅u),dΩ) + ∫ₚ(v⊙(∇(u)'⋅du),dΩ)
dc_t(μ,t,(u,p),(dut,dpt),(v,q)) = nothing

res_lin(μ,t,(u,p),(v,q)) = ∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
jac_lin(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
jac_lin_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)

res(μ,t,(u,p),(v,q)) = res_lin(μ,t,(u,p),(v,q)) + c(μ,t,(u,p),(v,q))
jac(μ,t,(u,p),(du,dp),(v,q)) = jac_lin(μ,t,(u,p),(du,dp),(v,q)) + dc(μ,t,(u,p),(du,dp),(v,q))
jac_t = jac_lin_t

T = Float
reffe_u = ReferenceFE(lagrangian,VectorValue{2,T},order)
reffe_p = ReferenceFE(lagrangian,T,order-1)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["noslip","inlet"])
trial_u = TransientTrialPFESpace(test_u,[g0,g])
test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldPFESpace([test_u,test_p])
trial = TransientMultiFieldPFESpace([trial_u,trial_p])
feop_lin = AffineFEOperator(res_lin,jac_lin,jac_lin_t,pspace,trial,test)
feop_nlin = FEOperator(c,dc,dc_t,pspace,trial,test)
feop = TransientPFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))

nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
fesolver = ThetaMethod(nls,θ,dt)

ϵ = 1e-4
load_solutions = false
save_solutions = true
load_structures = false
save_structures = true
norm_style = [:l2,:l2]
compute_supremizers = true
nsnaps_state = 50
nsnaps_mdeim = 30
nsnaps_test = 10
st_mdeim = false
rbinfo = BlockRBInfo(test_path;ϵ,norm_style,compute_supremizers,nsnaps_state,
  nsnaps_mdeim,nsnaps_test,st_mdeim)

# Offline phase
printstyled("OFFLINE PHASE\n";bold=true,underline=true)
if load_solutions
  sols,params = load(rbinfo,(BlockSnapshots{Vector{T}},Table))
else
  sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
  if save_solutions
    save(rbinfo,(sols,params,stats))
  end
end
if load_structures
  rbspace = load(rbinfo,BlockRBSpace{T})
  rbrhs,rblhs = load(rbinfo,(NTuple{2,BlockRBVecAlgebraicContribution{T}},
    NTuple{3,Vector{BlockRBMatAlgebraicContribution{T}}}),Ω)
else
  rbspace = reduced_basis(rbinfo,feop,sols)
  rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop_lin,feop_nlin,fesolver,rbspace,params)
  if save_structures
    save(rbinfo,(rbspace,rbrhs,rblhs))
  end
end

# Online phase
printstyled("ONLINE PHASE\n";bold=true,underline=true)
rb_solver(rbinfo,feop_lin,feop_nlin,fesolver,rbspace,rbrhs,rblhs,sols,params)
end
