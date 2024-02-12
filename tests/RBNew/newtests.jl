using Gridap
using Gridap.FESpaces
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.MultiField
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 1
dt = 0.1
t0 = 0.0
tf = 1.0

pranges = fill([0,1],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=3)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

########################## HEAT EQUATION ############################

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[7,8])
dΓn = Measure(Γn,degree)

a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6])
trial = TransientTrialParamFESpace(test,gμt)
feop = AffineTransientParamFEOperator(res,jac,jac_t,ptspace,trial,test,trian_res,trian_jac,trian_jac_t)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

dir = datadir(joinpath("heateq","toy_mesh"))
rbinfo = RBInfo(dir;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=5)

rbsolver = RBSolver(rbinfo,fesolver)

snaps,comp = RB.collect_solutions(rbsolver,feop,uh0μ)
red_op = reduced_operator(rbsolver,feop,snaps)
son = select_snapshots(snaps,6)
xrb = solve(rbsolver,red_op,ron)
STOP
# θ == 0.0 ? dtθ = dt : dtθ = dt*θ

# red_test = get_test(red_op)
# red_trial = get_trial(red_op)(ron)
# fe_trial = trial(ron)
# red_x = zero_free_values(red_trial)
# y = zero_free_values(fe_trial)
# z = similar(y)
# z .= 0.0

# ode_cache = allocate_cache(red_op,ron)
# nl_cache = nothing
# mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(red_op,ron,y,ode_cache)
# ode_cache = update_cache!(ode_cache,red_op,ron)
# A,b = ODETools._matrix_and_vector!(mat_cache,vec_cache,red_op,ron,dtθ,y,ode_cache,z)

# # jac_ad = get_values(red_op.lhs[1])[1]
# # xhF = (y,z)
# # fe_A,coeff_cache,lincomb_cache = mat_cache
# # # fe_sA = fe_matrix!(fe_A,red_op,ron,xhF,ode_cache)
# # red_times = RB._union_reduced_times(red_op)
# # red_r = ron[:,red_times]
# # A = fe_matrix!(fe_A,red_op.pop,red_r,xhF,(1,1),ode_cache)
# # LHS = get_values(A[1])[1]
# # ids_space = RB.get_indices_space(jac_ad)
# # ids_time = filter(!isnothing,indexin(red_times,RB.get_indices_time(jac_ad)))
# # ids_param = Base.OneTo(num_params(LHS))
# # snew = RB.reverse_snapshots_at_indices(LHS,ids_space)
# # zio = select_snapshots(snew,ids_time,ids_param)

# trial0 = trial(nothing)
# pA = ParamArray([assemble_matrix((u,v)->∫(a(μ,t)*∇(v)⋅∇(u))dΩ,trial0,test) for (μ,t) in ron])
# pM = ParamArray([assemble_matrix((u,v)->∫(v*u)dΩ,trial0,test) for (μ,t) in ron])
# snapsA = Snapshots(pA,ron)
# snapsM = Snapshots(pM,ron)
# Arb = RB.compress(red_trial,red_test,snapsA)
# Mrb = RB.compress(red_trial,red_test,snapsM;combine=(x,y) -> θ*(x-y))
# AMrb = Arb + Mrb

# errA = norm(A[1] - AMrb)

# μon = FEM._get_params(ron)[1]
# g_t(x,t) = μon[1]*exp(-x[1]/μon[2])*abs(sin(t/μon[3]))
# g_t(t) = x->g_t(x,t)
# fs_t = TransientTrialFESpace(test,g_t)
# dfs_t = ∂t(fs_t)
# R(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(μon,t)*∇(v)⋅∇(u))dΩ - ∫(f(μon,t)*v)dΩ - ∫(h(μon,t)*v)dΓn
# function get_res(t)
#   fs = fs_t(t)
#   dfs = dfs_t(t)
#   x0 = zeros(3)
#   xh = TransientCellField(EvaluationFunction(fs,x0),(EvaluationFunction(dfs,x0),))
#   assemble_vector(v->R(t,xh,v),test)
# end
# pR = ParamArray([get_res(t) for (μ,t) in ron])
# snapsR = Snapshots(pR,ron)
# Rrb = RB.compress(red_test,snapsR)

# errb = norm(b[1] + Rrb)

# xhF = (y,z)
# mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(red_op,ron,y,ode_cache)
# fe_b,coeff_cache,lincomb_cache = vec_cache
# red_times = RB._union_reduced_times(red_op)
# red_r = ron[:,red_times]
# red_B = fe_vector!(fe_b,red_op.pop,red_r,xhF,ode_cache) # this is wrong
# redBs = sum(get_values(red_B))

# rhs_ad = get_values(red_op.rhs)[2]
# # rhs_ad.integration_domain.indices_space
# cell_dof_ids = get_cell_dof_ids(test,Γn)
# indices_space_rows = fast_index(rhs_ad.integration_domain.indices_space,num_free_dofs(test))
# red_integr_cells = RB.get_reduced_cells(indices_space_rows,cell_dof_ids)
# red_trian = view(Γn,red_integr_cells)
# red_meas = Measure(red_trian,2)
# pH = ParamArray([assemble_vector(v->∫(h(μ,t)*v)red_meas,test) for (μ,t) in ron])
# sH = Snapshots(pH,ron)

# # new
# new_red_jac = affine_contribution()
# vals = get_values(red_op.lhs[1])
# new_red_jac[Ω] = vals[1]
# new_red_jac_t = affine_contribution()
# vals = get_values(red_op.lhs[2])
# new_red_jac_t[Ω] = vals[1]
# new_red_res = affine_contribution()
# vals = get_values(red_op.rhs)
# new_red_res[Ω] = vals[1]
# new_red_res[Γn] = vals[2]
# new_red_op = ReducedOperator(pop,(new_red_jac,new_red_jac_t),new_red_res)
# new_ode_cache = allocate_cache(new_red_op,ron)
# new_mat_cache,new_vec_cache = ODETools._allocate_matrix_and_vector(new_red_op,ron,y,new_ode_cache)
# new_fe_b,_,_ = new_vec_cache
# new_ode_cache = update_cache!(new_ode_cache,new_red_op,ron)
# red_times = RB._union_reduced_times(new_red_op)
# red_ron = ron[:,red_times]
# B = fe_vector!(new_fe_b,new_red_op.pop,red_ron,(y,z),new_ode_cache)
# Bs = sum(get_values(B))
# stop
# # r = realization(ptspace,nparams=3)
# # c(μ,t,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
# # a(μ,t,du,v) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
# # m(μ,t,dut,v) = ∫(v*dut)dΩ
# # feop_ok = AffineTransientParamFEOperator(m,a,c,ptspace,trial,test)
# # sol_ok = solve(fesolver,feop,uh0μ,r)
# # sol = solve(fesolver,feop,uh0μ,r)

# # for (x,xok) in zip(sol,sol_ok)
# #   xh,_ = x
# #   xh_ok,_ = xok
# #   @check get_free_dof_values(xh) ≈ get_free_dof_values(xh_ok)
# #   @check xh.dirichlet_values ≈ xh_ok.dirichlet_values
# # end

# odeop = get_algebraic_operator(feop)
# pop = GalerkinProjectionOperator(odeop,red_trial,red_test)
# red_lhs,red_rhs = RB.reduced_matrix_vector_form(rbsolver,pop,snaps)
# # red_op = reduced_operator(pop,red_lhs,red_rhs)
# trians_lhs = map(get_domains,red_lhs)
# trians_rhs = get_domains(red_rhs)
# # new_pop = change_triangulation(pop,trians_lhs,trians_rhs)
# # new_feop = FEM.change_triangulation(FEM.get_fe_operator(pop),trians_lhs,trians_rhs)
# newtrian_res = FEM._order_triangulations(feop.trian_res,trians_rhs)
# newtrian_jac,newtrian_jac_t = FEM._order_triangulations.(feop.trian_jacs,trians_lhs)
# porder = FEM.get_polynomial_order(test)
# newres,newjacs... = FEM._set_triangulation(res,jac,jac_t,newtrian_res,newtrian_jac,newtrian_jac_t,porder)

# ###########
# xh = TransientCellField(zero(trial0),(zero(trial0),))
# feop_trian = change_triangulation(feop,[trian_jac,trian_jac_t],trian_res)

# pR = ParamArray([assemble_vector(v->feop.op.res(FEM.get_params(ron)[1],t,xh,v),test) for (_,t) in ron])
# snapsR = Snapshots(pR,ron)

# _pR = ParamArray([assemble_vector(v->feop_trian.op.res(FEM.get_params(ron)[1],t,xh,v),test) for (_,t) in ron])
# _snapsR = Snapshots(_pR,ron)

# __pR = ParamArray([assemble_vector(v->red_op.pop.feop.feop.op.res(FEM.get_params(ron)[1],t,xh,v),test) for (_,t) in ron])
# __snapsR = Snapshots(__pR,ron)
