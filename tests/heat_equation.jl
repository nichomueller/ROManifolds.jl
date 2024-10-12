using Gridap
using Test
using DrWatson
using ReducedOrderModels.FEM
using ReducedOrderModels.RB

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)

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
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=1,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver,dir=test_dir)
using Serialization
p = deserialize("/home/nicholasmueller/git_repos/Mabla.jl/params.txt")
t = deserialize("/home/nicholasmueller/git_repos/Mabla.jl/times.txt")
r = FEM.GenericTransientParamRealization(ParamRealization(p),t,0.0)
# fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
sol = solve(fesolver,feop,uh0μ,r)
odesol = sol.odesol
stats = @timed begin
  vals = collect(sol)
end
fesnaps = Snapshots(vals,r)
festats = ComputationalStats(stats,60)

rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

# POD-MDEIM error
pod_err,mdeim_error = RB.pod_mdeim_error(rbsolver,feop,rbop,fesnaps)

using Gridap.FESpaces
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
using LinearAlgebra

s = select_snapshots(fesnaps,51)
r = get_realization(s)
dtθ = dt*θ
FEM.shift_time!(r,dt*(θ-1))
red_trial = get_trial(rbop)(r)
fe_trial = get_fe_trial(rbop)(r)
red_x = zero_free_values(red_trial)
y = zero_free_values(fe_trial)
z = similar(y)
z .= 0.0

ode_cache = allocate_cache(rbop,r)
mat_cache,vec_cache = ODETools._allocate_matrix_and_vector(rbop,r,y,ode_cache)
LinearAlgebra.fillstored!(mat_cache,zero(eltype(mat_cache)))
fe_sA = fe_jacobians!(mat_cache,rbop,r,(y,z),(1.0,1/dtθ),ode_cache)

ad = rbop.lhs[1][1]
coeff = RB.mdeim_coeff(ad,fe_sA[1][1])

# # coefficient part
C,CR = copy.(RB.get_coeff_cache(ad))
mdeim_interpolation = ad.mdeim_interpolation
basis_time = ad.basis.basis_time

ns = num_reduced_space_dofs(ad)
nt = num_reduced_times(ad)
np = length(CR)

bvec = reshape(fe_sA[1][1],:,np)
ldiv!(C,mdeim_interpolation,bvec)
# #

# jacobian_mdeim_lincomb(ad,coeff)

time_prod_cache,lincomb_cache = RB.get_lincomb_cache(ad)
fill!(lincomb_cache,zero(eltype(lincomb_cache)))

basis_time = ad.basis.metadata
basis_space = ad.basis.basis_space

using Kronecker
i = 1
lci = lincomb_cache[i]
ci = coeff[i]
j = 1
for col in axes(basis_time,3)
  for row in axes(basis_time,2)
    time_prod_cache[row,col] = sum(basis_time[:,row,col].*ci[:,j])
  end
end
contrib = kronecker(time_prod_cache,basis_space[j])

# alternative
function _combine_basis_time(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix;kwargs...)
  map(a->_combine_basis_time(a,B,C;kwargs...),eachcol(A))
end

function _combine_basis_time(a::AbstractVector,B::AbstractMatrix,C::AbstractMatrix;combine=(x,y)->x)
  nt_row = size(C,2)
  nt_col = size(B,2)

  T = eltype(B)
  bt_proj = zeros(T,nt_row,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    bt_proj[it,jt] = sum(C[:,it].*a.*B[:,jt])
    bt_proj_shift[it,jt] = sum(C[2:end,it].*a[2:end].*B[1:end-1,jt])
  end

  combine(bt_proj,bt_proj_shift)
end

Φt_trial = get_basis_time(red_trial)
Φt_test = copy(Φt_trial)
Φt = get_basis_time(ad.basis)
Φt_red = _combine_basis_time(Φt,Φt_test,Φt_trial;combine=(x,y)->θ*x+(1-θ)*y)

Φst_red_1 = kronecker(Φt_red[1],basis_space[1])
Φst_red_2 = kronecker(Φt_red[2],basis_space[1])
approx_contrib = Φst_red_1*C[1]+Φst_red_2*C[3]
