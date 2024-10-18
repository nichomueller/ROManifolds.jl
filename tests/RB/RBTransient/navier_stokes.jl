using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, NonlinearSystemBlock, BiformBlock, BlockTriangularSolver

using ReducedOrderModels
import ReducedOrderModels.RBSteady: get_jacobian_reduction,get_residual_reduction

θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 10*dt

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

model_dir = datadir(joinpath("models","model_circle.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
add_tag_from_tags!(labels,"dirichlet",["inlet"])

order = 2
degree = 2*order+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

const Re = 1000.0
a(x,μ,t) = μ[1]/Re*exp((sin(t)+cos(t))/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 0.5
inflow(μ,t) = abs(1-cos(9*π*t/(5*60*dt))+μ[3]*sin(μ[2]*9*π*t/(5*60*dt))/100)
g_in(x,μ,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

function newsave(dir,contrib::ArrayContribution;label::String="")
  for (i,c) in enumerate(get_values(contrib))
    l = RBSteady._get_label(label,i)
    save(dir,c;label=l)
  end
end

function newsave(dir,contrib::TupOfArrayContribution;label::String="")
  for (i,c) in enumerate(contrib)
    l = RBSteady._get_label(label,i)
    newsave(dir,c;label=l)
  end
end

function newload(dir,trian::Tuple{Vararg{Triangulation}};label="")
  a = ()
  for (i,t) in enumerate(trian)
    l = RBSteady._get_label(label,i)
    ai = load_snapshots(dir;label=l)
    a = (a...,ai)
  end
  return Contribution(a,trian)
end

function newload(dir,trian::Tuple{Vararg{Tuple{Vararg{Triangulation}}}};label="")
  a = ()
  for (i,t) in enumerate(trian)
    l = RBSteady._get_label(label,i)
    a = (a...,newload(dir,t;label=l))
  end
  return a
end

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
  dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_jac,trian_jac_t)
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,
  trial,test,trian_res,trian_jac)
feop = LinNonlinTransientParamFEOperator(feop_lin,feop_nlin)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
diag_blocks  = [NonlinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0*p*q)dΩ,test_p,test_p)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
solver_u = LUSolver()
solver_p = LUSolver()
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=true)
nlsolver = GridapSolvers.NewtonSolver(solver;rtol=1e-10,maxiter=20)
odesolver = ThetaMethod(nlsolver,dt,θ)
lu_nlsolver = GridapSolvers.NewtonSolver(LUSolver();rtol=1e-10,maxiter=20)
lu_odesolver = ThetaMethod(lu_nlsolver,dt,θ)

tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=50,sketch=:sprn)
rbsolver = RBSolver(odesolver,state_reduction;nparams_res=30,nparams_jac=20,nparams_djac=1)

test_dir = datadir("navier-stokes")
create_dir(test_dir)

r = realization(feop;nparams=1)
fesnaps,festats = solution_snapshots(rbsolver,feop,r,xh0μ)

# println(festats)
# save(test_dir,fesnaps)

# op_lin = get_algebraic_operator(feop.op_linear)
# jacs_lin = jacobian_snapshots(rbsolver,op_lin,fesnaps)
# ress_lin = residual_snapshots(rbsolver,op_lin,fesnaps)

# # newsave(test_dir,jacs_lin;label="jac_lin")
# # newsave(test_dir,ress_lin;label="res_lin")

# op_nlin = get_algebraic_operator(feop.op_nonlinear)
# jacs_nlin = jacobian_snapshots(rbsolver,op_nlin,fesnaps)
# ress_nlin = residual_snapshots(rbsolver,op_nlin,fesnaps)

# # newsave(test_dir,jacs_nlin;label="jac_nlin")
# # newsave(test_dir,ress_nlin;label="res_nlin")

# xtest = select_snapshots(fesnaps,51:60)
# rtest = get_realization(xtest)

# for tol in (1e-2,1e-3,1e-4)
#   test_dir_tol = joinpath(test_dir,"tol_$(tol)")

#   state_reduction = TransientReduction(coupling,tol,energy;nparams=50,sketch=:sprn)
#   rbsolver = RBSolver(lu_odesolver,state_reduction;nparams_res=30,nparams_jac=20,nparams_djac=1)
#   jac_red = get_jacobian_reduction(rbsolver)
#   res_red = get_residual_reduction(rbsolver)

#   red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)

#   red_lhs_lin = reduced_jacobian(jac_red,red_trial,red_test,jacs_lin)
#   red_rhs_lin = reduced_residual(res_red,red_test,ress_lin)
#   trians_rhs_lin = get_domains(red_rhs_lin)
#   trians_lhs_lin = get_domains(red_lhs_lin)
#   new_op_lin = change_triangulation(op_lin,trians_rhs_lin,trians_lhs_lin)
#   rbop_lin = GenericRBOperator(new_op_lin,red_trial,red_test,red_lhs_lin,red_rhs_lin)

#   red_lhs_nlin = reduced_jacobian(jac_red,red_trial,red_test,jacs_nlin)
#   red_rhs_nlin = reduced_residual(res_red,red_test,ress_nlin)
#   trians_rhs_nlin = get_domains(red_rhs_nlin)
#   trians_lhs_nlin = get_domains(red_lhs_nlin)
#   new_op_nlin = change_triangulation(op_nlin,trians_rhs_nlin,trians_lhs_nlin)
#   rbop_nlin = GenericRBOperator(new_op_nlin,red_trial,red_test,red_lhs_nlin,red_rhs_nlin)

#   rbop = LinearNonlinearTransientRBOperator(rbop_lin,rbop_nlin)
#   save(test_dir_tol,rbop)
# end

using Gridap.Visualization
dir = datadir("plts")
U = trial_u(r)
createpvd(dir) do pvd
  for i in param_eachindex(r)
    file = dir*"/u$i"*".vtu"
    Ui = param_getindex(U,i)
    vi = fesnaps[1][:,i,1]
    uhi = FEFunction(Ui,vi)
    pvd[i] = createvtk(Ω,file,cellfields=["u"=>uhi])
  end
end

P1 = trial_p
createpvd(dir) do pvd
  for i in param_eachindex(r)
    file = dir*"/p$i"*".vtu"
    vi = fesnaps[2][:,i,1]
    phi = FEFunction(P1,vi)
    pvd[i] = createvtk(Ω,file,cellfields=["p"=>phi])
  end
end
