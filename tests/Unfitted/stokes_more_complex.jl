using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ROM

pranges = (1,10,-1,5,1,2)
pspace = ParamSpace(pranges)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 40
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = TProductModel(pmin,pmax,partition)
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
feop = LinearParamFEOperator(l,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

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

# Q = TProductFESpace(Ωbg,reffe_p;conformity=:H1)
# P = ParamTrialFESpace(Q)
# B = assemble_matrix((p,v) -> ∫(p*(∇⋅(v)))dΩbg.measure,P,test_u)
# Φu = get_basis(rbop.test[1])
# Φp = get_basis(rbop.test[2])

# det(B'*B)
# B̂ = Φu'*B*Φp
# sqrt(det(B̂'*B̂))

using ROM.RBSteady
using ROM.ParamDataStructures

# energy′((du,dp),(v,q)) = ∫(du⋅v)dΩbg.measure + ∫(∇(v)⊙∇(du))dΩbg.measure + ∫(dp*q)dΩbg.measure
# coupling′((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩbg.measure
# test_p′ = TProductFESpace(Ωbg,reffe_p;conformity=:H1)
# test′ = MultiFieldParamFESpace([test_u.space,test_p′.space];style=BlockMultiFieldStyle())
# trial′ = MultiFieldParamFESpace([test_u.space,test_p′.space];style=BlockMultiFieldStyle())
# fesnaps′ = flatten(fesnaps)
# state_reduction′ = SupremizerReduction(coupling′,1e-4,energy;nparams=100)
# norm_matrix′ = assemble_matrix(energy′,test′,test′)
# supr_op′ = assemble_matrix(coupling′,test′,test′)
# proj′ = RBSteady.reduced_basis(state_reduction′.reduction,fesnaps′,norm_matrix′)
# enrich!(state_reduction′,proj′,norm_matrix′,supr_op′)

# Φu′ = get_basis(proj′[1])
# Φp′ = get_basis(proj′[2])

# B̂′ = Φu′'*B*Φp′
# sqrt(det(B̂'*B̂))

# POD
energy′((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ
coupling′((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
test_u′ = FESpace(Ωact,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u′ = ParamTrialFESpace(test_u′,gμ)
test_p′ = FESpace(Ωact,reffe_p;conformity=:H1)
trial_p′ = ParamTrialFESpace(test_p′)
test′ = MultiFieldParamFESpace([test_u′,test_p′];style=BlockMultiFieldStyle())
trial′ = MultiFieldParamFESpace([trial_u′,trial_p′];style=BlockMultiFieldStyle())
feop′ = LinearParamFEOperator(l,a,pspace,trial′,test′,domains)

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
