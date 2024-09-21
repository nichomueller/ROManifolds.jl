using Gridap
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 0.15

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

model_dir = datadir(joinpath("models","model_circle_2d_quad.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet_noslip",["walls","walls_p","cylinders","cylinders_p"])

order = 2
degree = 2*(order)+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]*exp((sin(t)+cos(t))/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 0.5
inflow(μ,t) = abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100)
g_in(x,μ,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet_noslip","inlet"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_0,gμt_in])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
# test_p = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_stiffness,trian_mass)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0*p*q)dΩ,test_p,test_p)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
solver_u = LUSolver()
solver_p = LUSolver()
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-7,verbose=true)
odesolver = ThetaMethod(solver,dt,θ)
lu_odesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(odesolver,ϵ;nparams_state=50,nparams_res=50,nparams_jac=30,nparams_test=10)
lu_rbsolver = RBSolver(lu_odesolver,ϵ;nparams_state=50,nparams_res=50,nparams_jac=30,nparams_test=10)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("stokes","model_circle_2d_quad")))

# results = load_solve(rbsolver,feop,test_dir)

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats,cache = solve(lu_rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

save(test_dir,fesnaps)
# save(test_dir,rbop)
save(test_dir,results)

show(results.timer)
println(compute_speedup(results))

s1 = select_snapshots(results.sol[1],[51])
sa1 = select_snapshots(results.sol_approx[1],1)
e1 = abs.(s1 - sa1)
r1 = get_realization(s1)
U1 = trial_u(r1)

using Gridap.Visualization
dir = datadir("plts")
createpvd(dir) do pvd
  for i in param_eachindex(r1)
    file = dir*"/u$i"*".vtu"
    Ui = param_getindex(U1,i)
    vi = e1[:,i,1]
    uhi = FEFunction(Ui,vi)
    pvd[i] = createvtk(Ω,file,cellfields=["u"=>uhi])
  end
end

s1 = select_snapshots(results.sol[2],[51])
sa1 = select_snapshots(results.sol_approx[2],1)
e1 = abs.(s1 - sa1)
r1 = get_realization(s1)
P1 = trial_p

createpvd(dir) do pvd
  for i in param_eachindex(r1)
    file = dir*"/p$i"*".vtu"
    vi = e1[:,i,1]
    phi = FEFunction(P1,vi)
    pvd[i] = createvtk(Ω,file,cellfields=["p"=>phi])
  end
end

# using BlockArrays
# red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# basis = red_trial.basis
# X = assemble_norm_matrix(feop)
# s1 = select_snapshots(fesnaps[2],51:51)
# RBSteady.pod_error(red_test[1],s1,X[Block(1,1)])

# s1 = flatten_snapshots(s1)
# s2 = RBTransient.change_mode(s1)
# basis_space = get_basis_space(basis[2])
# basis_time = get_basis_time(basis[2])
# err_space = norm(Matrix(s1) - basis_space*basis_space'*X[Block(2,2)]*Matrix(s1)) / norm(s1)
# err_time = norm(Matrix(s2) - basis_time*basis_time'*Matrix(s2)) / norm(s2)
# B = assemble_coupling_matrix(feop)[Block(1,2)]

# bsu = get_basis_space(basis[1])
# bsp = get_basis_space(basis[2])
# B̂ = bsu'*B*bsp

# i = 47
# Ui = param_getindex(U1,i)
# vi = e1[:,i,1]
# uhi = FEFunction(Ui,vi)
# cv = get_cell_dof_values(uhi)
# cvn = lazy_map(norm,cv) |> collect

# wrong_cell_1 = 125
# wrong_dofs_1 = get_cell_dof_ids(test_u)[wrong_cell_1]
# wrong_dofs_1_free = wrong_dofs_1[wrong_dofs_1.>0]
# s1[wrong_dofs_1_free,:,1]
# sa1[wrong_dofs_1_free,:,1]

# wrong_dofs_1_dir = -wrong_dofs_1[wrong_dofs_1.<0]
# s1[wrong_dofs_1_dir,:,1]
# sa1[wrong_dofs_1_dir,:,1]
