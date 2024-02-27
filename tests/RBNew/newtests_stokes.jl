using Gridap
using Gridap.FESpaces
using GridapGmsh
using ForwardDiff
using BlockArrays
using LinearAlgebra
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.Fields
using Gridap.MultiField
using BlockArrays
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
model_dir = datadir(joinpath("meshes","perforated_plate.json"))
model = DiscreteModelFromFile(model_dir)
# domain = (0,1,0,1)
# partition = (20,20)
# model = CartesianDiscreteModel(domain, partition)
# labels = get_face_labeling(model)
# add_tag_from_tags!(labels,"diri1",[7])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = 1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = 1-cos(2π*t/tf)+sin(μ[2]*2π*t/tf)/μ[1]
g_in(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_w(x,μ,t) = VectorValue(0.0,0.0)
g_w(μ,t) = x->g_w(x,μ,t)
gμt_w(μ,t) = TransientParamFunction(g_w,μ,t)
g_c(x,μ,t) = VectorValue(0.0,0.0)
g_c(μ,t) = x->g_c(x,μ,t)
gμt_c(μ,t) = TransientParamFunction(g_c,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
jac(μ,t,u,(du,dp),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ + ∫(q*(∇⋅(du)))dΩ
jac_t(μ,t,u,(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
# test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["diri1"])
# trial_u = TransientTrialParamFESpace(test_u,gμt)
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w,gμt_c])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("stokes","perforated_plate"))
info = RBInfo(dir;norm_style=[:l2,:l2],nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20,
  st_mdeim=true,compute_supremizers=true,variable_name=("vel","press"))

rbsolver = RBSolver(info,fesolver)

snaps,comp = ode_solutions(rbsolver,feop,xh0μ)
red_op = reduced_operator(rbsolver,feop,snaps)

son = select_snapshots(snaps,RB.online_params(info))
ron = get_realization(son)
xrb,comprb = solve(rbsolver,red_op,ron)
son_rev = reverse_snapshots(son)
RB.space_time_error(son_rev,xrb)

info_space = RBInfo(dir;norm_style=[:l2,:l2],nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20,
  compute_supremizers=true,variable_name=("vel","press"))
rbsolver_space = RBSolver(info_space,fesolver)
red_op_space = reduced_operator(rbsolver_space,feop,snaps)
xrb_space, = solve(rbsolver_space,red_op_space,ron)
RB.space_time_error(son_rev,xrb_space)

results = rb_results(rbsolver,red_op,snaps,xrb,comp,comprb)
# results = solve(rbsolver,feop,xh0μ)
generate_plots(rbsolver,feop,results)


# generate_plots(rbsolver,feop,results)
plt_dir = joinpath(info.dir,"plots")
fe_plt_dir = joinpath(plt_dir,"fe_solution")
# _plot(trial,sol;dir=fe_plt_dir,varname=r.name)
free_values = get_values(sol)
r = get_realization(sol)
trials = trial(r)
sh = FEFunction(trials,free_values)
nfields = length(trials.spaces)
# _plot(sh[1],r,varname=:vel)
trian = get_triangulation(sh[1])
rt = FEM.get_at_time(r,:initial)
dt = FEM.get_delta_time(r)
create_dir(dir)
createpvd(rt,dir)
FEM.shift_time!(rt,dt)
files = ParamString(dir,rt)
solh_t = FEM._getindex(sh[1],1)
vtk = createvtk(trian,files,cellfields=["vel"=>solh_t])

bs,bt = red_op.op.trial.basis_space[2],red_op.op.trial.basis_time[2]

M = select_snapshots(son[2],51) |> collect
errs = norm(M - bs*bs'*M) / norm(M)
errt = norm(M' - bt*bt'*M') / norm(M')

B = assemble_matrix((p,v) -> ∫(p*(∇⋅(v)))dΩ,trial_p,test_u)
Bs,Bt = red_op.op.trial.basis_space[1],red_op.op.trial.basis_time[1]

C = Bs'*B*bs
sqrt(det(C'*C))

soff = select_snapshots(snaps,RB.offline_params(info))
bases = reduced_basis(soff;ϵ=RB.get_tol(info))
basis_space = bases[1]
basis_primal,basis_dual = basis_space.array
supr_i = B * basis_dual
gram_schmidt!(supr_i,basis_primal)
supr_basis_primal = hcat(basis_primal,supr_i)

cc = supr_basis_primal'*B*basis_dual
sqrt(det(C'*C))
Uc,Sc,Vc = svd(C)

dd = supr_i'*B*basis_dual
det(dd)

for i = axes(basis_primal,2)
  C = basis_primal[:,i]'*B*basis_dual
  println(sqrt(abs(det(C'*C))))
end
for i = axes(supr_i,2)
  C = supr_i[:,i]'*B*basis_dual
  println(sqrt(abs(det(C'*C))))
end

r = get_realization(snaps)
tr = trial_u(r)
dr = tr.dirichlet_values

gh = interpolate_dirichlet(gμt(get_params(r),get_times(r)),tr)
x = get_cell_points(Ω)
dghdx = (∇⋅(gh))(x)
for i = eachindex(dghdx)
  if sum(sum(dghdx[i])) != 0.0
    println(i)
  end
end
