using Gridap
using GridapEmbedded
using ROM

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

R = 0.5
L = 0.8*(2*R)
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(L,0.0)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R
dp = pmax - pmin

n = 20
partition = (n,n)
bgmodel = TProductDiscreteModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo)
Ωbg = Triangulation(bgmodel)
Ω = Triangulation(cutgeo,PHYSICAL)
Ωact = Triangulation(cutgeo,ACTIVE)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

nΓ = get_normal_vector(Γ)
nΓg = get_normal_vector(Γg)

const γd = 10.0
const γg = 0.1
const h = dp[1]/n

ν(μ) = x->sum(μ)
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

g(μ) = x->μ[3]*sum(x)
gμ(μ) = ParamFunction(g,μ)

# non-symmetric formulation

a(μ,u,v,dΩ,dΓ) = ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ - ∫( νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
b(μ,u,v,dΩ,dΓ) = a(μ,u,v,dΩ,dΓ) - ∫( v*fμ(μ) )dΩ - ∫( νμ(μ)*(nΓ⋅∇(v))*gμ(μ) )dΓ
domains = FEDomains((Ω,Γ),(Ω,Γ))

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(bgmodel,geo)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TProductFESpace(Ωact,Ωbg,bgcell_to_inoutcut,reffe;conformity=:H1)
trial = ParamTrialFESpace(test,gμ)
feop = LinearParamFEOperator(b,a,pspace,trial,test,domains)

tol = 1e-4
energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
state_reduction = TTSVDReduction(tol,energy;nparams=100)
fesolver = LinearFESolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction)

# offline
fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

# online
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)

# test
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

using ROM.RBSteady
rbsnaps = RBSteady.to_snapshots(rbop.trial,x̂,μon)

v = x[:,:,1]
x̂ = project(rbop.test.subspace,v)

using DrWatson
using ROM.ParamDataStructures
r1 = get_realization(x)[1]
S1 = get_param_data(x)[1]
Ŝ1 = get_param_data(rbsnaps)[1]
plt_dir = datadir("plts")
create_dir(plt_dir)
uh1 = FEFunction(param_getindex(trial(r1),1),S1)
ûh1 = FEFunction(param_getindex(trial(r1),1),Ŝ1)
writevtk(Ω,joinpath(plt_dir,"sol.vtu"),cellfields=["uhapp"=>ûh1,"uh"=>uh1,"eh"=>uh1-ûh1])
writevtk(Ωact,joinpath(plt_dir,"solact.vtu"),cellfields=["uhapp"=>ûh1,"uh"=>uh1,"eh"=>uh1-ûh1])

STOP
# act_model = get_active_model(Ωact)
# bg_model = get_background_model(Ωact)
# act_f = CartesianFESpace(act_model,reffe;trian=Ωact,conformity=:H1)
# bg_f = CartesianFESpace(bg_model,reffe;conformity=:H1)

# using ROM.DofMaps
# bg_cell_to_bg_cellin = bgcell_to_inoutcut
# bg_odof_to_act_odof = DofMaps._get_bg_odof_to_act_odof(bg_f,act_f)
# agg_dof_to_bg_dof = DofMaps._get_bg_cutout_dofs(bg_f,act_f,bg_cell_to_bg_cellin)

# trian_a = get_triangulation(act_f)
# D = num_cell_dims(trian_a)
# glue = get_glue(trian_a,Val(D))
# acell_to_bg_cell = glue.tface_to_mface
# acell_to_bg_cellin = collect(lazy_map(Reindex(bg_cell_to_bg_cellin),acell_to_bg_cell))
# bg_ndofs = num_free_dofs(bg_f)
# acell_to_dof_ids = get_cell_dof_ids(bg_f)

# using GridapEmbedded.Interfaces

# function _mytest(bg_ndofs,bg_cellids,bg_cell_to_inoutcut)
#   bg_dof_to_mask = fill(false,bg_ndofs)
#   touched = fill(false,bg_ndofs)
#   bg_cellids = Table(bg_cellids)
#   bg_dof_to_cells = lazy_map(_DofToCell(bg_cellids),1:bg_ndofs)
#   dof_cache = array_cache(bg_cellids)
#   cell_cache = array_cache(bg_dof_to_cells)
#   for (bg_cell,inoutcut) in enumerate(bg_cell_to_inoutcut)
#     if inoutcut == OUT
#       bg_dofs = getindex!(dof_cache,bg_cellids,bg_cell)
#       for dof in bg_dofs
#         if !touched[dof]
#           touched[dof] = true
#           bg_dof_to_mask[dof] = true
#         end
#       end
#     elseif inoutcut == CUT
#       bg_dofs = getindex!(dof_cache,bg_cellids,bg_cell)
#       for dof in bg_dofs
#         if !touched[dof]
#           bg_cells = getindex!(cell_cache,bg_dof_to_cells,dof)
#           mark = true
#           for cell in bg_cells
#             if bg_cell_to_inoutcut[cell] == IN
#               mark = false
#               break
#             end
#           end
#           if mark
#             touched[dof] = true
#             bg_dof_to_mask[dof] = true
#           end
#         end
#       end
#     end
#   end
#   bg_dof_to_mask
# end

# bg_ndofs = num_free_dofs(bg_f)
# bg_cellids = Table(get_cell_dof_ids(bg_f))
# bg_cell_to_inoutcut = compute_bgcell_to_inoutcut(bgmodel,geo)
# bg_dof_to_mask = _mytest(bg_ndofs,bg_cellids,bg_cell_to_inoutcut)

# cellcut = 6
# bg_dofs = bg_cellids[cellcut]

# struct _DofToCell{A} <: Map
#   cellids::A
# end

# function Arrays.return_cache(k::_DofToCell,dof::Int)
#   array_cache(k.cellids)
# end

# function Arrays.evaluate!(cache,k::_DofToCell,dof::Int)
#   cells = Int32[]
#   for cell in 1:length(k.cellids)
#     cell_dofs = getindex!(cache,k.cellids,cell)
#     if dof ∈ cell_dofs
#       append!(cells,cell)
#     end
#   end
#   cells
# end

# d2c = _DofToCell(bg_cellids)
