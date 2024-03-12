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

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.5

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 50
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
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test)
feop = FEOperatorWithTrian(_feop,trian_res,trian_jac,trian_jac_t)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = TTRBSolver(fesolver,ϵ;nsnaps_state=24,nsnaps_test=1,nsnaps_mdeim=1)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_test")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)


################################################################################
s = select_snapshots(fesnaps,51)
feA,feb = RB._jacobian_and_residual(fesolver,feop,s)
rbA,rbb = RB._jacobian_and_residual(rbsolver,rbop.op,s)

RB.interpolation_error(rbop.lhs[1][1],feA[1][1],rbA[1][1])
RB.interpolation_error(rbop.lhs[2][1],feA[2][1],rbA[2][1])
RB.interpolation_error(rbop.rhs[1],feb[1],rbb[1])
RB.interpolation_error(rbop.rhs[2],feb[2],rbb[2])

### MDEIM
soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)

son = select_snapshots(fesnaps,51)
ron = get_realization(son)
θ == 0.0 ? dtθ = dt : dtθ = dt*θ

r = copy(ron)
FEM.shift_time!(r,dt*(θ-1))

rb_trial = red_trial(r)
fe_trial = trial(r)
red_x = zero_free_values(rb_trial)
y = zero_free_values(fe_trial)
z = similar(y)
z .= 0.0

op = get_algebraic_operator(feop)
ode_cache = allocate_cache(op,r)
ode_cache = update_cache!(ode_cache,op,r)

A,b = ODETools._allocate_matrix_and_vector(op,r,y,ode_cache)
ODETools._matrix_and_vector!(A,b,op,r,dt*θ,y,ode_cache,z)

B = red_trial.basis.basis_spacetime
n_red = size(B,2)
n_space_dofs = num_free_dofs(test)

A1 = A[1][1]
BDA1 = RB.BDiagonal(getproperty.(A1.array,:values))
A1_red = θ*B'*(BDA1*B)
for n = 2:num_times(r)
  m = n-1
  A1_n = A1[n]
  B_n = B[(n-1)*n_space_dofs+1:n*n_space_dofs,:]
  B_m = B[(m-1)*n_space_dofs+1:m*n_space_dofs,:]
  A1_red += (1-θ)*B_n'*A1_n*B_m
end

M = A[2][1]
BDM = RB.BDiagonal(getproperty.(M.array,:values))
M_red = θ*B'*(BDM*B)
for n = 2:num_times(r)
  m = n-1
  M_n = M[n]
  B_n = B[(n-1)*n_space_dofs+1:n*n_space_dofs,:]
  B_m = B[(m-1)*n_space_dofs+1:m*n_space_dofs,:]
  M_red -= θ*B_n'*M_n*B_m
end

# alternative:
# RB.compress_combine_basis_space_time(
#   RB.VecOfBDiagonalSparseMat2Mat([BDA1]),BB,CC,B_shift,C_shift;combine)[1]

ss = select_snapshots(fesnaps,51)
pop = RB.PODOperator(get_algebraic_operator(feop),red_trial,red_test)
contribs_mat,contribs_vec = RB.jacobian_and_residual(rbsolver,pop,ss)

# this means that compress_combine_basis_space_time works
# AA = RB.get_basis_spacetime(basis)
# BB = RB.get_basis_spacetime(red_trial.basis)
# CC = RB.get_basis_spacetime(red_test.basis)
# B_shift = RB._shift(BB,1:num_times(red_test.basis)-1,RB.num_space_dofs(red_test.basis))
# C_shift = RB._shift(CC,2:num_times(red_test.basis),RB.num_space_dofs(red_test.basis))
# metadata = RB.compress_combine_basis_space_time(AA,BB,CC,B_shift,C_shift;combine)

# @assert A1_red ≈ RB.compress_combine_basis_space_time(
#   RB.VecOfBDiagonalSparseMat2Mat([BDA1]),BB,CC,B_shift,C_shift;combine)[1]

# stiffness
sA = contribs_mat[1][1]
basis = reduced_basis(sA)
combine = (x,y) -> fesolver.θ*x+(1-fesolver.θ)*y
proj_basis = RB.compress_basis(basis,red_trial,red_test;combine)

# prelim check
norm(sA[:] - basis.basis_spacetime*basis.basis_spacetime'*sA[:]) / norm(sA[:])
norm(Snapshots(A1,ron) - sA)

# check mdeim
A1_mdeim = basis.basis_spacetime*coeffA[1]
A1_mdeim + RB.VecOfBDiagonalSparseMat2Mat([BDA1]) # here too, there is a wrong - sign

# check mdeim reduced structure
ii = rbop.lhs[1][1].integration_domain
ids_space,ids_time = ii.indices_space,ii.indices_time
A_rev = reverse_snapshots(contribs_mat[1][1])
A_at_ids = RB.select_snapshots_entries(A_rev,ids_space,ids_time)
coeffA = ldiv!(zeros(1),rbop.lhs[1][1].mdeim_interpolation,A_at_ids[1])
A1_mdeim_red = proj_basis.metadata[1]*coeffA[1]
A1_mdeim_red + A1_red # why not - ???

# # wrong, this means that coeffA is wrong
# # red trian is wrong
# smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
# pop = RB.PODOperator(get_algebraic_operator(feop),red_trial,red_test)
# contribs_mat,contribs_vec = RB.jacobian_and_residual(rbsolver,pop,smdeim)
# S = contribs_mat[1][1]
# b = reduced_basis(S;ϵ=RB.get_tol(rbsolver))
# indices_spacetime = get_mdeim_indices(b.basis_spacetime)
# indices_space = RB.fast_index(indices_spacetime,RB.num_space_dofs(b))
# indices_time = RB.slow_index(indices_spacetime,RB.num_space_dofs(b))
# lu_interp = lu(view(b.basis_spacetime,indices_spacetime,:))
# recast_indices_space = RB.recast_indices(b.basis_spacetime,indices_space)
# integration_domain = ReducedIntegrationDomain(recast_indices_space,indices_time)
# red_trian = reduce_triangulation(Ω,integration_domain,red_trial,red_test)
# red_meas = Measure(red_trian,2)
# A = [assemble_matrix((u,v)->∫(a(μ,t)*∇(v)⋅∇(u))dΩ,trial(nothing),test) for (μ,t) in ron]
# red_A = [assemble_matrix((u,v)->∫(a(μ,t)*∇(v)⋅∇(u))red_meas,trial(nothing),test) for (μ,t) in ron]

# mass
sM = contribs_mat[2][1]
basis = reduced_basis(sM)
combine = (x,y) -> fesolver.θ*(x-y)

# check space-time reduction
AA = RB.VecOfBDiagonalSparseMat2Mat([BDM])
BB = RB.get_basis_spacetime(red_trial.basis)
CC = RB.get_basis_spacetime(red_test.basis)
B_shift = RB._shift(BB,RB.num_space_dofs(red_test.basis),:backwards)
C_shift = RB._shift(CC,RB.num_space_dofs(red_test.basis),:forwards)
@assert M_red ≈ RB.compress_combine_basis_space_time(AA,BB,CC,B_shift,C_shift;combine)[1]

# check mdeim
M_mdeim = basis.basis_spacetime*coeffM[1]
BDM = RB.BDiagonal(getproperty.(A[2][1].array,:values))
M_mdeim + RB.VecOfBDiagonalSparseMat2Mat([BDM])

# check mdeim reduced structure
proj_basis = RB.compress_basis(basis,red_trial,red_test;combine)
ii = rbop.lhs[2][1].integration_domain
ids_space,ids_time = ii.indices_space,ii.indices_time
M_rev = reverse_snapshots(contribs_mat[2][1])
M_at_ids = RB.select_snapshots_entries(M_rev,ids_space,ids_time)
coeffM = ldiv!(zeros(1),rbop.lhs[2][1].mdeim_interpolation,M_at_ids[1])
M_mdeim_red = proj_basis.metadata[1]*coeffM[1]
M_mdeim_red - M_red # why not - ???
