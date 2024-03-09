using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.FESpaces
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using Test
using DrWatson
using Mabla.FEM
using Mabla.RB
using LinearAlgebra
using SparseArrays

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 5
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

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
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"],vector_type=TTVector{1,Float64})
trial = TransientTrialParamFESpace(test,gμt)
_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test)
feop = FEOperatorWithTrian(_feop,trian_res,trian_jac,trian_jac_t)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = TTRBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_test")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver,dir=test_dir)

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)

soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
# norm_matrix = assemble_norm_matrix(feop)
# cores = RB.ttsvd(soff)

# epsilon = [ϵ,ϵ,ϵ]
# opt = Opt([30,30,30],epsilon)
# tt = TensorTrain(soff;opt)

# norm(tt.cores[1]-cores[1])
# norm(tt.cores[2]-cores[2])
# norm(tt.cores[3]-cores[3])

# basis_st = RB.get_basis_spacetime(cores)
# b12 = basis(tt,1,2) |> base2mat
# norm(basis_st-b12)

# testing general algorithm

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)

# ## proj test --> works
# cores = RB.ttsvd(soff)
# B = RB.get_basis_spacetime(cores)
# x = reshape(son,200,1)
# norm(x - B*B'*x) / norm(x)
# ##

pop = RB.PODOperator(get_algebraic_operator(feop),red_trial,red_test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
contribs_mat,contribs_vec = RB.jacobian_and_residual(rbsolver,pop,smdeim)
red_mat = RB.reduced_matrix_form(rbsolver,pop,contribs_mat)
red_vec = RB.reduced_vector_form(rbsolver,pop,contribs_vec)
rbop = RB.PODMDEIMOperator(pop,red_mat,red_vec)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)

ii = red_mat[1].values[1].integration_domain
ids_space,ids_time = ii.indices_space,ii.indices_time
s = contribs_mat[1][1]
srev = reverse_snapshots(s)
ssrev = select_snapshots(srev,ids_space,ids_time,1:10)
ciao
# Affine Decomposition part
# basis_mat_1 = reduced_basis(contribs_mat[1][1];ϵ)
# # lu_interp,integration_domain = RB.mdeim(rbsolver,basis_mat_1)
# b = basis_mat_1
# indices_spacetime = get_mdeim_indices(b.basis_spacetime)
# indices_space = fast_index(indices_spacetime,RB.num_space_dofs(b))
# indices_time = slow_index(indices_spacetime,RB.num_space_dofs(b))
# lu_interp = lu(view(b.basis_spacetime,indices_spacetime,:))
# recast_indices_space = RB.recast_indices(b.basis_spacetime,indices_space)
# integration_domain = ReducedIntegrationDomain(recast_indices_space,indices_time)

# combine = (x,y) -> θ*x+(1-θ)*y
# A = RB.get_basis_spacetime(basis_mat_1)
# B = RB.get_basis_spacetime(red_trial.basis)
# C = RB.get_basis_spacetime(red_test.basis)
# B_shift = RB.shift(B,1:num_times(red_test.basis)-1,RB.num_space_dofs(red_test.basis))
# C_shift = RB.shift(C,2:num_times(red_test.basis),RB.num_space_dofs(red_test.basis))
# metadata = RB.compress_combine_basis_space_time(A,B,C,B_shift,C_shift;combine)

# these tests work!
# basis_mat_1 = reduced_basis(contribs_mat[1][1];ϵ)
# basis_mat_2 = reduced_basis(contribs_mat[2][1];ϵ)

# basis_vec_1 = reduced_basis(contribs_vec[1];ϵ)
# basis_vec_2 = reduced_basis(contribs_vec[2];ϵ)

# son = select_snapshots(fesnaps,1)
# ron = get_realization(son)
# θ == 0.0 ? dtθ = dt : dtθ = dt*θ

# r = copy(ron)
# FEM.shift_time!(r,dt*(θ-1))

# rb_trial = red_trial(r)
# fe_trial = trial(r)
# red_x = zero_free_values(rb_trial)
# y = zero_free_values(fe_trial)
# z = similar(y)
# z .= 0.0

# op = get_algebraic_operator(feop)
# ode_cache = allocate_cache(op,r)
# ode_cache = update_cache!(ode_cache,op,r)

# A,b = ODETools._allocate_matrix_and_vector(op,r,y,ode_cache)
# ODETools._matrix_and_vector!(A,b,op,r,dt*θ,y,ode_cache,z)

# B = red_trial.basis.basis_spacetime
# n_red = size(B,2)
# n_space_dofs = num_free_dofs(test)

# # ## proj test
# # B = basis_vec_1.basis_spacetime
# # sb = Snapshots(b[1],ron)
# # x = reshape(sb,200,1)
# # norm(x - B*B'*x) / norm(x)
# # ## proj test
# # B = basis_vec_2.basis_spacetime
# # sb = Snapshots(b[2],ron)
# # x = reshape(sb,200,1)
# # norm(x - B*B'*x) / norm(x)
# # ## proj test
# # B = basis_mat_1.basis_spacetime
# # A1 = A[1][1]
# # A1_red = zeros(4,4)
# # for n = 1:num_times(r)
# #   A1_n = A1[n]
# #   B_n = B[(n-1)*n_space_dofs+1:n*n_space_dofs,:]
# #   A1_red += B_n'*A1_n*B_n
# # end
# # ## proj test

# A1 = A[1][1]
# BDA1 = RB.BDiagonal(getproperty.(A1.array,:values))
# A1_red = θ*B'*(BDA1*B)
# for n = 2:num_times(r)
#   m = n-1
#   A1_n = A1[n]
#   B_n = B[(n-1)*n_space_dofs+1:n*n_space_dofs,:]
#   B_m = B[(m-1)*n_space_dofs+1:m*n_space_dofs,:]
#   A1_red += (1-θ)*B_n'*A1_n*B_m
# end

# A2 = A[2][1]
# A2_red = zeros(4,4)
# for n = 1:num_times(r)
#   A2_n = A2[n]
#   B_n = B[(n-1)*n_space_dofs+1:n*n_space_dofs,:]
#   A2_red += θ*B_n'*A2_n*B_n
# end
# for n = 2:num_times(r)
#   m = n-1
#   A2_n = A2[n]
#   B_n = B[(n-1)*n_space_dofs+1:n*n_space_dofs,:]
#   B_m = B[(m-1)*n_space_dofs+1:m*n_space_dofs,:]
#   A2_red -= θ*B_n'*A2_n*B_m
# end

# b1 = b[1]
# sb1 = Snapshots(b1,ron)
# b1_red = B'*reshape(sb1,:,1)

# b2 = b[2]
# sb2 = Snapshots(b2,ron)
# b2_red = B'*reshape(sb2,:,1)

# A_red = A1_red+A2_red
# b_red = b1_red+b2_red
# u_red = A_red \ b_red
# u_rec = B*u_red

# U_rec = reshape(u_rec,n_space_dofs,:)

#############################

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test)
feop = FEOperatorWithTrian(_feop,trian_res,trian_jac,trian_jac_t)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_test")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(feop,rbsolver,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
