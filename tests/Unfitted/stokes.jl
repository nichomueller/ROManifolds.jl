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
n = 20
partition = (n,n)

geo = !disk(R,x0=Point(0.5,0.5))

bgmodel = TProductDiscreteModel(pmin,pmax,partition)
labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])

cutgeo = cut(bgmodel,geo)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
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
  ∫( - v⋅(n_Γ⋅∇(u))*νμ(μ) + (n_Γ⋅∇(v))⋅u*νμ(μ) + (p*n_Γ)⋅v + (q*n_Γ)⋅u )dΓ
)

l(μ,(u,p),(v,q),dΩ) = ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ

trian_res = (Ω,)
trian_jac = (Ω,Γ)
domains = FEDomains(trian_res,trian_jac)

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg  + ∫(dp*q)dΩbg + ∫(∇(v)⊙∇(du))dΩbg

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(bgmodel,geo)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TProductFESpace(Ωact,Ωbg,bgcell_to_inoutcut,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TProductFESpace(Ωact,Ωbg,bgcell_to_inoutcut,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = fill(1e-4,4)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100,unsafe=true)
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

reffe_u1 = ReferenceFE(lagrangian,Float64,order)
Vall = OrderedFESpace(bgmodel.model,reffe_u1;conformity=:H1,dirichlet_tags="dirichlet")

in_dof_map = Float64.(vec(get_internal_dof_map(feop)[1][:,:,1]))
in_dof_map[findall(!iszero,in_dof_map)] .= 1
uhin = FEFunction(Vall,in_dof_map)

dof_map = Float64.(vec(get_dof_map(feop)[1][:,:,1]))
dof_map[findall(!iszero,dof_map)] .= 1
uh = FEFunction(Vall,dof_map)

writevtk(Ωbg.trian,joinpath(plt_dir,"sol.vtu"),cellfields=["uhin"=>uhin,"uh"=>uh])

using ROManifolds.RBSteady
rbsnaps = RBSteady.to_snapshots(rbop.trial,x̂,ronline)

V = OrderedFESpace(bgmodel.model,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
Q = OrderedFESpace(bgmodel.model,reffe_p;conformity=:H1)
Y = MultiFieldParamFESpace([V,Q];style=BlockMultiFieldStyle())
X = assemble_matrix(energy,Y,Y)
Xu = X[Block(1,1)]
v = vec(x[1][:,:,:,1])
v̂ = vec(rbsnaps[1][:,:,:,1])
ev = v-v̂
sqrt(sum(ev'*Xu*ev)) / sqrt(sum(v'*Xu*v))

in_dof_map = get_internal_dof_map(feop)
inv_in_dof_map = invert(in_dof_map[1])
vin = vec(change_dof_map(x[1],in_dof_map[1])[:,:,:,1])
v̂in = vec(change_dof_map(rbsnaps[1],inv_in_dof_map)[:,:,:,1])
evin = vin-v̂in
sqrt(sum(evin'*Xu*evin)) / sqrt(sum(vin'*Xu*vin))
