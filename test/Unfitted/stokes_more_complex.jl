using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ROManifolds

pdomain = (1,10,-1,5,1,2)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 40
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = TProductDiscreteModel(pmin,pmax,partition)
labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])

cutgeo = cut(bgmodel,geo2)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ωact_out = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)

n_Γ = get_normal_vector(Γ)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΩ_out = Measure(Ω_out,degree)

ν(μ) = x -> μ[1]
νμ(μ) = ParamFunction(ν,μ)

g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = ParamFunction(g,μ)

f(μ) = x -> VectorValue(0.0,0.0)
fμ(μ) = ParamFunction(f,μ)

g_0(μ) = x -> VectorValue(0.0,0.0)
gμ_0(μ) = ParamFunction(g_0,μ)

a(μ,(u,p),(v,q),dΩ,dΩ_out,dΓ) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
  ∫( ∇(v)⊙∇(u) )dΩ_out +
  ∫( - v⋅(n_Γ⋅∇(u))*νμ(μ) + (n_Γ⋅∇(v))⋅u*νμ(μ) + (p*n_Γ)⋅v + (q*n_Γ)⋅u )dΓ
)

l(μ,(u,p),(v,q),dΩ) = ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ

trian_res = (Ω,)
trian_jac = (Ω,Ω_out,Γ)
domains = FEDomains(trian_res,trian_jac)

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg + ∫(∇(v)⊙∇(du))dΩbg + ∫(dp*q)dΩbg

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TProductFESpace(Ωact,Ωbg,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TProductFESpace(Ωact,Ωbg,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamOperator(l,a,pspace,trial,test,domains)

fesolver = LUSolver()

tol = fill(1e-4,4)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

dir = datadir("stokes_ttsvd_temp_temp")
create_dir(dir)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,ronline)

save(dir,fesnaps)
save(dir,rbop)

x,festats = solution_snapshots(rbsolver,feop,ronline)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,ronline)

V = TProductFESpace(Ωbg,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
Q = TProductFESpace(Ωbg,reffe_p;conformity=:H1)
P = ParamTrialFESpace(Q)
B = assemble_matrix((p,v) -> ∫(p*(∇⋅(v)))dΩbg.measure,P,V)
Φu = get_basis(rbop.test[1])
Φp = get_basis(rbop.test[2])

det(B'*B)
B̂ = Φu'*B*Φp
sqrt(det(B̂'*B̂))

using ROManifolds.RBSteady
using ROManifolds.ParamDataStructures

# energy′((du,dp),(v,q)) = ∫(du⋅v)dΩbg.measure + ∫(∇(v)⊙∇(du))dΩbg.measure + ∫(dp*q)dΩbg.measure
# coupling′((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩbg.measure
# test_u′ = TProductFESpace(Ωbg,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
# test_p′ = TProductFESpace(Ωbg,reffe_p;conformity=:H1)
# test′ = MultiFieldParamFESpace([test_u.space,test_p′.space];style=BlockMultiFieldStyle())
# trial′ = MultiFieldParamFESpace([test_u.space,test_p′.space];style=BlockMultiFieldStyle())
# fesnaps′ = flatten(fesnaps)
# state_reduction′ = SupremizerReduction(coupling′,1e-4,energy;nparams=100)
# norm_matrix′ = assemble_matrix(energy′,test′,test′)
# supr_op′ = assemble_matrix(coupling′,test′,test′)
# proj′ = RBSteady.reduced_basis(state_reduction′.reduction,fesnaps′,norm_matrix′)
# enrich!(state_reduction′,proj′,norm_matrix′,supr_op′)
B′ = assemble_matrix((p,v) -> ∫(p*(∇⋅(v)))dΩ,test_p.space,test_u.space)
proj′ = rbop′.test.subspace
Φu′ = get_basis(proj′[1])
Φp′ = get_basis(proj′[2])

B̂′ = Φu′'*B′*Φp′
sqrt(det(B̂'*B̂))

# POD
energy′((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ
coupling′((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
test_u′ = FESpace(Ωact,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u′ = ParamTrialFESpace(test_u′,gμ)
test_p′ = FESpace(Ωact,reffe_p;conformity=:H1)
trial_p′ = ParamTrialFESpace(test_p′)
test′ = MultiFieldParamFESpace([test_u′,test_p′];style=BlockMultiFieldStyle())
trial′ = MultiFieldParamFESpace([trial_u′,trial_p′];style=BlockMultiFieldStyle())
feop′ = LinearParamOperator(l,a,pspace,trial′,test′,domains)

state_reduction′ = SupremizerReduction(coupling′,1e-4,energy′;nparams=100)
rbsolver′ = RBSolver(fesolver,state_reduction′;nparams_res=50,nparams_jac=50)

dir′ = datadir("stokes_pod_temp_temp")
create_dir(dir′)

fesnaps′ = Snapshots(get_param_data(fesnaps),get_dof_map(test′),get_realization(fesnaps))
rbop′ = reduced_operator(rbsolver′,feop′,fesnaps′)
x̂′,rbstats′ = solve(rbsolver′,rbop′,ronline)
x′ = Snapshots(get_param_data(x),get_dof_map(test′),get_realization(x))
perf′ = eval_performance(rbsolver′,feop′,rbop′,x′,x̂′,festats,rbstats′,ronline)

save(dir′,fesnaps′)
save(dir′,rbop′)

# rbsnaps = RBSteady.to_snapshots(RBSteady.get_trial(rbop),x̂,ronline)
V = TProductFESpace(Ωbg,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
U = ParamTrialFESpace(V,gμ)
P = TProductFESpace(Ωbg,reffe_p;conformity=:H1)

r1 = get_realization(x)[1]
S1 = get_param_data(x[1])[1]
Ŝ1 = get_param_data(rbsnaps[1])[1]
plt_dir = datadir("plts")
create_dir(plt_dir)
uh1 = OrderedFEFunction(param_getindex(trial_u(r1),1),S1)
ûh1 = OrderedFEFunction(param_getindex(U(r1),1),Ŝ1)
writevtk(Ω,joinpath(plt_dir,"sol.vtu"),cellfields=["uh"=>uh1,"ûh"=>ûh1,"eh"=>ûh1-uh1])

S2 = get_param_data(x[2])[1]
Ŝ2 = get_param_data(rbsnaps[2])[1]
uh2 = OrderedFEFunction(test_p,S2)
ûh2 = OrderedFEFunction(P,Ŝ2)
writevtk(Ω,joinpath(plt_dir,"sol_p.vtu"),cellfields=["ph"=>uh2,"p̂h"=>ûh2,"eh"=>ûh2-uh2])

# compute proj errors
# RBSteady.projection_error(rbop.trial,get_param_data(x),ronline)
V = TProductFESpace(Ωbg,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
P = TProductFESpace(Ωbg,reffe_p;conformity=:H1)
Xu = assemble_matrix((u,v) -> ∫(u⋅v)dΩbg.measure + ∫(∇(v)⊙∇(u))dΩbg.measure,V.space,V.space)
Us = reshape(collect(fesnaps[1]),:,100)
error_u = Us - Φu*Φu'*Xu*Us

Xp = assemble_matrix((u,v) -> ∫(u⋅v)dΩbg.measure,P.space,P.space)
Ps = reshape(collect(fesnaps[2]),:,100)
error_p = Ps - Φp*Φp'*Xp*Ps

# orth?
norm_matrix = assemble_matrix(feop,energy)
basis_u = reduced_basis(state_reduction.reduction,fesnaps[1],norm_matrix[1,1])
BU = get_basis(basis_u)
BU'*Xu*BU
Xu11 = Xu[1:6320,1:6320]
kron(norm_matrix[1,1])

Xok = assemble_matrix((u,v) -> ∫(u⋅v)dΩbg.measure + ∫(∇(v)⊙∇(u))dΩbg.measure,V.space.space,V.space.space)

Xuok = kron(I(2),Xu11)
BU'*Xuok*BU
