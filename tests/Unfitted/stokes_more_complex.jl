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
n = 20
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

a(μ,(u,p),(v,q)) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
  ∫( ∇(v)⊙∇(u) )dΩ_out +
  ∫( - v⋅(n_Γ⋅∇(u))*νμ(μ) + (n_Γ⋅∇(v))⋅u*νμ(μ) + (p*n_Γ)⋅v + (q*n_Γ)⋅u )dΓ
)

l(μ,(u,p),(v,q)) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
  ∫( ∇(v)⊙∇(gμ_0(μ)) )dΩ_out +
  ∫( (n_Γ⋅∇(v))⋅gμ_0(μ)*νμ(μ) + (q*n_Γ)⋅gμ_0(μ) )dΓ
)

trian_res = (Ω,Ω_out,Γ)
trian_jac = (Ω,Ω_out,Γ)
domains = FEDomains(trian_res,trian_jac)

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg + ∫(∇(v)⊙∇(du))dΩbg + ∫(dp*q)dΩbg

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TProductFESpace(Ωbg,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TProductFESpace(Ωact,Ωbg,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test)

fesolver = LinearFESolver(LUSolver())

tol = fill(1e-4,4)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,ronline)

x,festats = solution_snapshots(rbsolver,feop,ronline)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,ronline)


x, = solution_snapshots(rbsolver,feop;nparams=1)
u1 = flatten_snapshots(x[1])[:,1]
p1 = flatten_snapshots(x[2])[:,1]
r1 = get_realization(x[1])[1]
U1 = param_getindex(trial_u(r1),1)
P1 = trial_p(nothing)
uh = FEFunction(U1,u1)
ph = FEFunction(P1,p1)
writevtk(Ω,datadir("plts/sol"),cellfields=["uh"=>uh,"ph"=>ph])
writevtk(Ωbg.trian,datadir("plts/sol_bg"),cellfields=["uh"=>uh,"ph"=>ph])

V = FESpace(Ωbg.trian,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
U = ParamTrialFESpace(V,gμ)
Q = FESpace(Ωact,reffe_p;conformity=:H1)
P = ParamTrialFESpace(Q)
Y = MultiFieldParamFESpace([V,Q];style=BlockMultiFieldStyle())
X = MultiFieldParamFESpace([U,P];style=BlockMultiFieldStyle())
ffeop = LinearParamFEOperator(l,a,pspace,X,Y)
xx, = solution_snapshots(rbsolver,ffeop;nparams=1)

using Gridap.FESpaces
using Gridap.Algebra
using ROM.ParamSteady

r = realization(feop)
UU = trial_u(r)
uh = zero(UU)
x = get_free_dof_values(uh)
op = get_algebraic_operator(feop)
nlop = ParamNonlinearOperator(op,r)

b = residual(nlop,x)
A = jacobian(nlop,x)

##############################################################
UUU = U(r)
uuh = zero(UUU)
xx = get_free_dof_values(uuh)
oop = get_algebraic_operator(ffeop)
nnlop = ParamNonlinearOperator(oop,r)

bb = residual(nnlop,x)
AA = jacobian(nnlop,x)

# UNIVARIATE

au(μ,u,v) = (∫( νμ(μ)*∇(v)⊙∇(u) )dΩ + ∫( ∇(v)⊙∇(u) )dΩ_out + ∫( - v⋅(n_Γ⋅∇(u))*νμ(μ) + (n_Γ⋅∇(v))⋅u*νμ(μ) )dΓ)
lu(μ,u,v) = (∫( νμ(μ)*∇(v)⊙∇(u) )dΩ +∫( (n_Γ⋅∇(v))⋅gμ_0(μ)*νμ(μ) )dΓ)

au_simple(μ,u,v) = ∫( νμ(μ)*∇(v)⊙∇(u) )dΩ #+ ∫( ∇(v)⊙∇(u) )dΩ_out
lu_simple(μ,u,v) = ∫( νμ(μ)*∇(v)⊙∇(u) )dΩ

feop_u = LinearParamFEOperator(lu_simple,au_simple,pspace,trial_u,test_u)
UU = trial_u(r)
uh = zero(UU)
fill!(uh.free_values,1.0)
x = get_free_dof_values(uh)
op = get_algebraic_operator(feop_u)
nlop = ParamNonlinearOperator(op,r)

b = residual(nlop,x)
A = jacobian(nlop,x)

ffeop = LinearParamFEOperator(lu_simple,au_simple,pspace,U,V)
UUU = U(r)
uuh = zero(UUU)
fill!(uuh.free_values,1.0)
xx = get_free_dof_values(uuh)
oop = get_algebraic_operator(ffeop)
nnlop = ParamNonlinearOperator(oop,r)

bb = residual(nnlop,x)
AA = jacobian(nnlop,x)

norm(param_getindex(b,1)) ≈ norm(param_getindex(bb,1))
norm(param_getindex(A,1)) ≈ norm(param_getindex(AA,1))

assem = SparseMatrixAssembler(trial_u,test_u)
vecdata = collect_cell_vector(test_u,lu_simple(r,uh,get_fe_basis(test_u)))

v1 = nz_counter(get_vector_builder(assem),(get_rows(assem),))
symbolic_loop_vector!(v1,assem,vecdata)
v2 = nz_allocation(v1)
# numeric_loop_vector!(v2,assem,vecdata)
cellvec = vecdata[1][1]
cellids = vecdata[2][1]
rows_cache = array_cache(cellids)
vals_cache = array_cache(cellvec)
add! = FESpaces.AddEntriesMap(+)
cell = 312
rows = getindex!(rows_cache,cellids,cell)
vals = getindex!(vals_cache,cellvec,cell)
evaluate!(nothing,add!,v2,vals,rows)

#############
aassem = SparseMatrixAssembler(U,V)
vvecdata = collect_cell_vector(V,lu_simple(r,uuh,get_fe_basis(V)))

vv1 = nz_counter(get_vector_builder(aassem),(get_rows(aassem),))
symbolic_loop_vector!(vv1,aassem,vvecdata)
vv2 = nz_allocation(vv1)
# numeric_loop_vector!(v2,assem,vecdata)
ccellvec = vvecdata[1][1]
ccellids = vvecdata[2][1]
rrows_cache = array_cache(ccellids)
vvals_cache = array_cache(ccellvec)
rrows = getindex!(rrows_cache,ccellids,cell)
vvals = getindex!(vvals_cache,ccellvec,cell)
evaluate!(nothing,add!,vv2,vvals,rrows)

v = get_fe_basis(test_u)
cf = ∇(uh)#∇(v)⊙∇(uh)
x = get_cell_points(Ω)
cfx = cf(x)

vv = get_fe_basis(V)
ccf = ∇(uuh)#∇(vv)⊙∇(uuh)
ccfx = ccf(x)
