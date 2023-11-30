module HeatEquationTest

using Gridap
using Mabla

root = pwd()
model = DiscreteModelFromFile(joinpath(root,"models/elasticity_3cyl2D.json"))
test_path = "$root/results/HeatEquation/elasticity_3cyl2D"
order = 1
degree = 2*order
Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

ranges = fill([1.,10.],3)
pspace = PSpace(ranges)

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

res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

T = Float
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = PTTrialFESpace(test,g)
feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)
t0,tf,dt,θ = 0.,0.3,0.005,0.5
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

ϵ = 1e-4
load_solutions = true
save_solutions = true
load_structures = false
save_structures = true
postprocess = true
norm_style = :H1
nsnaps_state = 50
nsnaps_mdeim = 20
nsnaps_test = 10
st_mdeim = true
rbinfo = RBInfo(test_path;ϵ,norm_style,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)

# Offline phase
printstyled("OFFLINE PHASE\n";bold=true,underline=true)
if load_solutions
  sols,params = load(rbinfo,(Snapshots{Vector{T}},Table))
else
  sols,params,stats = collect_solutions(rbinfo,fesolver,feop)
  if save_solutions
    save(rbinfo,(sols,params,stats))
  end
end
if load_structures
  rbspace = load(rbinfo,RBSpace{T})
  rbrhs,rblhs = load(rbinfo,(RBVecAlgebraicContribution{T},
    Vector{RBMatAlgebraicContribution{T}}),Ω,Γn)
else
  rbspace = reduced_basis(rbinfo,feop,sols)
  rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
  if save_structures
    save(rbinfo,(rbspace,rbrhs,rblhs))
  end
end
# Online phase
printstyled("ONLINE PHASE\n";bold=true,underline=true)
results = rb_solver(rbinfo,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)
if postprocess
  plot_results(rbinfo,feop,fesolver,results)
  save(rbinfo,results)
end
end

nsnaps_test = rbinfo.nsnaps_test
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)

# l2
rbinfo = RBInfo(test_path;ϵ,norm_style=:l2,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)
rbspace = reduced_basis(rbinfo,feop,sols)
rbrhs,rblhs = collect_compress_rhs_lhs(rbinfo,feop,fesolver,rbspace,params)
cache,(_,lhs) = Gridap.ODEs.TransientFETools.allocate_cache(op,rbspace)
stats = @timed begin
  rhs,(_lhs,_lhs_t) = collect_rhs_lhs_contributions!(cache,rbinfo,op,rbrhs,rblhs,rbspace)
  @. lhs = _lhs+_lhs_t
  rb_snaps_test = Mabla.RB.rb_solve(fesolver.nls,rhs,lhs)
end
approx_snaps_test = recast(rb_snaps_test,rbspace)
sol_approx = Mabla.RB.space_time_matrices(approx_snaps_test;nparams=nsnaps_test)
k = 60
norm(sol_approx[1][:,k] - snaps_test[k])
norm(snaps_test[k])

# H1
_rbinfo = RBInfo(test_path;ϵ,norm_style=:H1,nsnaps_state,nsnaps_mdeim,nsnaps_test,st_mdeim)
_rbspace = reduced_basis(_rbinfo,feop,sols)
_rbrhs,_rblhs = collect_compress_rhs_lhs(_rbinfo,feop,fesolver,_rbspace,params)
cache,(_,_lhs) = Gridap.ODEs.TransientFETools.allocate_cache(op,_rbspace)
stats = @timed begin
  _rhs,(__lhs,__lhs_t) = collect_rhs_lhs_contributions!(cache,_rbinfo,op,_rbrhs,_rblhs,_rbspace)
  @. _lhs = __lhs+__lhs_t
  _rb_snaps_test = Mabla.RB.rb_solve(fesolver.nls,_rhs,_lhs)
end
_approx_snaps_test = recast(_rb_snaps_test,_rbspace)
_sol_approx = Mabla.RB.space_time_matrices(_approx_snaps_test;nparams=nsnaps_test)
k = 60
norm_matrix = get_norm_matrix(_rbinfo,feop)
norm(_sol_approx[1][:,k] - snaps_test[k],norm_matrix)
norm(_sol_approx[1][:,k] - snaps_test[k])
norm(snaps_test[k],norm_matrix)

norm(_bs*_bs'*snaps_test[k] - snaps_test[k],norm_matrix)
norm(_bs*_bs'*snaps_test[k] - snaps_test[k])
norm(_bs*_bs'*norm_matrix*snaps_test[k] - snaps_test[k],norm_matrix)
norm(_bs*_bs'*norm_matrix*snaps_test[k] - snaps_test[k])
norm(bs*bs'*snaps_test[k] - snaps_test[k])

norm(_sol_approx[1][:,k] - snaps_test[k],L)
norm(snaps_test[k],L)
