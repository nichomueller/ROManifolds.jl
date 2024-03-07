using Gridap
using Gridap.FESpaces
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using Gridap.Helpers
using Gridap.Fields
using Test
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.02

n = 4
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
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
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ + ∫(gμt(μ,t)*v*du)dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

T = Float64
reffe = ReferenceFE(lagrangian,T,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test)
feop = FEOperatorWithTrian(_feop,trian_res,trian_jac,trian_jac_t)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=10,nsnaps_test=2,nsnaps_mdeim=2)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","toy_mesh")))
fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

s1 = select_snapshots(fesnaps,1)
intp_err = RB.interpolation_error(rbsolver,feop,rbop,s1)
proj_err = RB.linear_combination_error(rbsolver,feop,rbop,s1)

feA,feb = RB._jacobian_and_residual(RB.get_fe_solver(rbsolver),feop,s1)
# rbA,rbb = RB._jacobian_and_residual(rbsolver,rbop,s1)

function get_fe_snaps(_r)
  # full order matrix
  r = copy(_r)
  FEM.shift_time!(r,dt*(θ-1))
  trial0 = trial(nothing)
  pA = ParamArray([assemble_matrix((u,v)->∫(a(μ,t)*∇(v)⋅∇(u))dΩ,trial0,test) for (μ,t) in r])
  pM = ParamArray([assemble_matrix((u,v)->∫(v*u)dΩ,trial0,test)/(θ*dt) for (μ,t) in r])
  snapsA = Snapshots(pA,r)
  snapsM = Snapshots(pM,r)

  # full order vector

  function get_res(μ,t)
    g_t(x,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
    g_t(t) = x->g_t(x,t)
    fs_t = TransientTrialFESpace(test,g_t)
    dfs_t = ∂t(fs_t)
    fs = fs_t(t)
    dfs = dfs_t(t)

    R(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ - ∫(h(μ,t)*v)dΓn

    x0 = zeros(num_free_dofs(test))
    xh = TransientCellField(EvaluationFunction(fs,x0),(EvaluationFunction(dfs,x0),))
    assemble_vector(v->R(t,xh,v),test)
  end
  pR = ParamArray([get_res(μ,t) for (μ,t) in r])
  snapsR = Snapshots(pR,r)

  return snapsA,snapsM,snapsR
end

snapsA,snapsM,snapsR = get_fe_snaps(get_realization(s1))
snapsR ≈ feb[1] + feb[2]
snapsRrb = RB.compress(snapsR,get_test(rbop))

op = get_algebraic_operator(feop)
θ == 0.0 ? dtθ = dt : dtθ = dt*θ
r = copy(get_realization(s1))
FEM.shift_time!(r,dt*(θ-1))
ode_cache = allocate_cache(op,r)
w0 = get_values(s1)
w0 .= 0.0
vθ = similar(w0)
vθ .= 0.0
ode_cache = update_cache!(ode_cache,op,r)
nlop = ThetaMethodParamOperator(op,r,dtθ,w0,ode_cache,vθ)
b = allocate_residual(nlop,w0)
# residual!(b,nlop,w0)
uF = nlop.u0
vθ = nlop.vθ
# residual!(b,nlop.odeop,nlop.r,(uF,vθ),nlop.ode_cache)
Xh, = ode_cache
xh=TransientCellField(EvaluationFunction(Xh[1],uF),(EvaluationFunction(Xh[2],vθ),))
# residual!(b,nlop.odeop.feop,nlop.r,xh,nlop.ode_cache)
v = get_fe_basis(test)
dc = nlop.odeop.feop.op.res(get_params(r),get_times(r),xh,v)
assem = FEM.get_param_assembler(nlop.odeop.feop.op.assem,r)
BOH = map(b.values,nlop.odeop.feop.trian_res) do btrian,trian
  vecdata = FEM.collect_cell_vector_for_trian(test,dc,trian)
  assemble_vector!(btrian,assem,vecdata)
end

μ = FEM._get_params(r)[1]
t = 0.01
g_t(x,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g_t(t) = x->g_t(x,t)
fs_t = TransientTrialFESpace(test,g_t)
dfs_t = ∂t(fs_t)
fs = fs_t(t)
dfs = dfs_t(t)

R(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ - ∫(h(μ,t)*v)dΓn

x0 = zeros(num_free_dofs(test))
_xh = TransientCellField(EvaluationFunction(fs,x0),(EvaluationFunction(dfs,x0),))
assemble_vector(v->R(t,_xh,v),test)

dc_ok = R(t,_xh,v)

lazy_getter(a,i=1) = lazy_map(x->getindex(x,i),a)
dc_ok[Γn] ≈ lazy_getter(dc[Γn])
dc_ok[Ω] ≈ lazy_getter(dc[Ω])

pod_err = RB.pod_error(get_trial(rbop),fesnaps,assemble_norm_matrix(feop))
mdeim_err = RB.mdeim_error(rbsolver,feop,rbop,fesnaps)

s1 = select_snapshots(fesnaps,1)
intp_err = RB.interpolation_error(rbsolver,feop,rbop,s1)
# proj_err = RB.linear_combination_error(rbsolver,feop,rbop,fesnaps)

feA,feb = RB._jacobian_and_residual(RB.get_fe_solver(rbsolver),feop,s1)
feA_comp = RB.compress(rbsolver,feA,get_trial(rbop),get_test(rbop))
feb_comp = RB.compress(rbsolver,feb,get_test(rbop))
rbA,rbb = RB._jacobian_and_residual(rbsolver,rbop,s1)
errA = RB._rel_norm(feA_comp,rbA)
errb = RB._rel_norm(feb_comp,rbb)

function get_res_snaps(_r)
  # full order matrix
  r = copy(_r)
  FEM.shift_time!(r,dt*(θ-1))

  function get_res(μ,t)
    g_t(x,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
    g_t(t) = x->g_t(x,t)
    fs_u_t = TransientTrialFESpace(test_u,g_t)
    dfs_u_t = ∂t(fs_u_t)
    fs = MultiFieldFESpace([fs_u_t(t),trial_p];style=BlockMultiFieldStyle())
    dfs = MultiFieldFESpace([dfs_u_t(t),trial_p];style=BlockMultiFieldStyle())

    R(t,(u,p),(v,q)) = ∫(v⋅∂t(u))dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ

    x0 = zero_free_values(fs)
    xh = TransientCellField(EvaluationFunction(fs,x0),(EvaluationFunction(dfs,x0),))
    assemble_vector(v->R(t,xh,v),test)
  end

  pR = ParamArray([get_res(μ,t) for (μ,t) in r])
  snapsR = Snapshots(pR,r)

  return snapsR
end

snapsR = get_res_snaps(r)

M1 = feb[1][1]
_M1 = collect(snapsR)[1:760,:]
norm(M1-_M1)


# wrong
op = get_algebraic_operator(feop)
θ == 0.0 ? dtθ = dt : dtθ = dt*θ
r = get_realization(s1)
FEM.shift_time!(r,dt*(θ-1))
ode_cache = allocate_cache(op,r)
w0 = get_values(s1)
w0 .= 0.0
vθ = similar(w0)
vθ .= 0.0
ode_cache = update_cache!(ode_cache,op,r)
nlop = ThetaMethodParamOperator(op,r,dtθ,w0,ode_cache,vθ)
b = allocate_residual(nlop,w0)
# residual!(b,nlop,w0)
Xh, = ode_cache
xh=TransientCellField(EvaluationFunction(Xh[1],w0),(EvaluationFunction(Xh[2],vθ),))
# residual!(b,op.feop,r,xh,ode_cache)
newop = FEM._remove_saddle_point_operator(op.feop)

# right
μ = FEM._get_params(r)[1]
t = dt
g_t(x,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
g_t(t) = x->g_t(x,t)
fs_u_t = TransientTrialFESpace(test_u,g_t)
dfs_u_t = ∂t(fs_u_t)
fs = MultiFieldFESpace([fs_u_t(t),trial_p];style=BlockMultiFieldStyle())
dfs = MultiFieldFESpace([dfs_u_t(t),trial_p];style=BlockMultiFieldStyle())

R(t,(u,p),(v,q)) = ∫(v⋅∂t(u))dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ

x0 = zero_free_values(fs)
xh_ok = TransientCellField(EvaluationFunction(fs,x0),(EvaluationFunction(dfs,x0),))
v = get_fe_basis(test)
dc_ok = R(t,xh_ok,v)




red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
odeop = get_algebraic_operator(feop)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
# contribs_mat,contribs_vec = RB.jacobian_and_residual(rbsolver,pop,smdeim)

# red_op_lin = reduced_operator(rbsolver,RB.get_linear_operator(pop),fesnaps)
pop = RB.PODOperator(odeop,red_trial,red_test)
oplin = RB.get_linear_operator(pop)
contribs_mat,contribs_vec = RB.jacobian_and_residual(rbsolver,oplin,smdeim)
A = contribs_mat[1][1]
A11 = A[1,1]

# bs,bt = RB.reduced_basis(A11)
basis_space,basis_time = RB.compute_bases(A11)
BSR = RB.recast(A11,basis_space)
