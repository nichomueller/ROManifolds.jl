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
n = 30
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])

cutgeo = cut(bgmodel,geo2)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ωact_out = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Γ = EmbeddedBoundary(cutgeo)

n_Γ = get_normal_vector(Γ)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

ν(μ) = x -> μ[1]
νμ(μ) = ParamFunction(ν,μ)

g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = ParamFunction(g,μ)

f(μ) = x -> VectorValue(0.0,0.0)
fμ(μ) = ParamFunction(f,μ)

g_0(μ) = x -> VectorValue(0.0,0.0)
gμ_0(μ) = ParamFunction(g_0,μ)

a(μ,(u,p),(v,q),dΩ,dΓ) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
  ∫( - v⋅(n_Γ⋅∇(u))*νμ(μ) + (n_Γ⋅∇(v))⋅u*νμ(μ) + (p*n_Γ)⋅v - (q*n_Γ)⋅u )dΓ
)

l(μ,(u,p),(v,q),dΩ) = ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ

trian_res = (Ω,)
trian_jac = (Ω,Γ)
domains = FEDomains(trian_res,trian_jac)

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ
coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = FESpace(Ωact,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = FESpace(Ωact,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,ronline)

x,festats = solution_snapshots(rbsolver,feop,ronline)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,ronline)

# x, = solution_snapshots(rbsolver,feop;nparams=1)

##########################

bgmodel′ = TProductDiscreteModel(pmin,pmax,partition)
labels = get_face_labeling(bgmodel′)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])

cutgeo′ = cut(bgmodel′,geo2)

Ωbg′ = Triangulation(bgmodel′)
Ωact′ = Triangulation(cutgeo′,ACTIVE)
Ω′ = Triangulation(cutgeo′,PHYSICAL_IN)
Γ′ = EmbeddedBoundary(cutgeo′)

n_Γ′ = get_normal_vector(Γ′)

order = 2
degree = 2*order

dΩbg′ = Measure(Ωbg′,degree)
dΩ′ = Measure(Ω′,degree)
dΓ′ = Measure(Γ′,degree)

a′(μ,(u,p),(v,q),dΩ′,dΓ′) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ′ +
  ∫( - v⋅(n_Γ′⋅∇(u))*νμ(μ) + (n_Γ′⋅∇(v))⋅u*νμ(μ) + (p*n_Γ′)⋅v + (q*n_Γ′)⋅u )dΓ′
)

l′(μ,(u,p),(v,q),dΩ′) = ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ′

trian_res′ = (Ω′,)
trian_jac′ = (Ω′,Γ′)
domains′ = FEDomains(trian_res′,trian_jac′)

coupling′((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩbg′ + ∫(dp*∂₂(v))dΩbg′
energy′((du,dp),(v,q)) = ∫(du⋅v)dΩbg′ + ∫(∇(v)⊙∇(du))dΩbg′ + ∫(dp*q)dΩbg′

test_u′ = TProductFESpace(Ωact′,Ωbg′,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u′ = ParamTrialFESpace(test_u′,gμ)
test_p′ = TProductFESpace(Ωact′,Ωbg′,reffe_p;conformity=:H1)
trial_p′ = ParamTrialFESpace(test_p′)
test′ = MultiFieldParamFESpace([test_u′,test_p′];style=BlockMultiFieldStyle())
trial′ = MultiFieldParamFESpace([trial_u′,trial_p′];style=BlockMultiFieldStyle())
feop′ = LinearParamFEOperator(l′,a′,pspace,trial′,test′,domains′)

state_reduction′ = SupremizerReduction(coupling′,fill(1e-4,4),energy′;nparams=100)
rbsolver′ = RBSolver(fesolver,state_reduction′;nparams_res=50,nparams_jac=50)

fesnaps′, = solution_snapshots(rbsolver′,feop′)
rbop′ = reduced_operator(rbsolver′,feop′,fesnaps′)
x̂′,rbstats′ = solve(rbsolver′,rbop′,ronline)
x′,festats′ = solution_snapshots(rbsolver′,feop′,ronline)
perf′ = eval_performance(rbsolver′,feop′,rbop′,x′,x̂′,festats′,rbstats′,ronline)

plt_dir = datadir("plts")
create_dir(plt_dir)

r1 = get_realization(x′)[1]
S1 = get_param_data(x′[1])[1]

uh1 = OrderedFEFunction(param_getindex(trial_u′(r1),1),S1)
# writevtk(Ω′,joinpath(plt_dir,"sol.vtu"),cellfields=["uh"=>uh1])
# visualization_data(Ω′,joinpath(plt_dir,"sol.vtu");cellfields=["uh"=>uh1])
trian = Ω′
cellfields = ["uh"=>uh1]
ref_grids = map((reffe) -> UnstructuredGrid(reffe), get_reffes(trian))
visgrid = Visualization.VisualizationGrid(trian,ref_grids)

cdata = Visualization._prepare_cdata(Dict(),visgrid.sub_cell_to_cell)
pdata = Visualization._prepare_pdata(trian,cellfields,visgrid.cell_to_refpoints)

using Gridap.CellData
x = CellPoint(visgrid.cell_to_refpoints,trian,ReferenceDomain())
_v = CellField(uh1,trian)
evaluate(_v,x)
