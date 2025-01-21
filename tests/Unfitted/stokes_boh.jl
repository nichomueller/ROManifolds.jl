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

ν(x,μ) = μ[1]
ν(μ) = x->ν(x,μ)
νμ(μ) = ParamFunction(ν,μ)

g(x,μ) = VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)

f(x,μ) = VectorValue(0.0,0.0)
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g_0(x,μ) = VectorValue(0.0,0.0)
g_0(μ) = x->g_0(x,μ)
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
# test_u = TProductFESpace(Ωbg,reffe_u;conformity=:H1)
test_u = FESpace(Ωbg.trian,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
# test_p = TProductFESpace(Ωact,Ωbg,reffe_p;conformity=:H1)
test_p = FESpace(Ωact,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test)

fesolver = LinearFESolver(LUSolver())

tol = fill(1e-4,4)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

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

V = TProductFESpace(Ωbg,reffe_u;conformity=:H1)
Q = TProductFESpace(Ωact,Ωbg,reffe_p;conformity=:H1)
QQ = TProductFESpace(Ωbg,reffe_p;conformity=:H1)

sdof_map = get_sparse_dof_map(V,QQ)
change_domain(sdof_map)

using Gridap.CellData
dof_map_rows′ = change_domain(get_dof_map(QQ),Ωact)
dof_map_cols′ = change_domain(get_dof_map(V),Ωact)
sparsity′ = change_domain(get_sparsity(sdof_map),get_dof_map(QQ),get_dof_map(V))
get_sparse_dof_map(trial,test,sparsity′)

matrix = SparsityPattern(V,V).sparsity.matrix

using Gridap
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.ReferenceFEs
using ROM
using ROM.Utils
using ROM.DofMaps
using ROM.TProduct

pmin = Point(0,0)
pmax = Point(1,1)
n = 3
partition = (n,n)
model = TProductModel(pmin,pmax,partition)
trian = Triangulation(model)
Ω = trian.trian
dΩ = Measure(Ω,2)
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
space = FESpace(model.model,reffe;conformity=:H1,dirichlet_tags=[1,2,5])
space′ = NewFESpace(trian,reffe;conformity=:H1,dirichlet_tags=[1,2,5])

# fe_dof_basis = get_data(get_fe_dof_basis(space))
# orders = get_polynomial_orders(space)
# trian = get_triangulation(space)
# model = get_background_model(trian)
# tface_to_mface = get_tface_to_mface(trian)
# desc = get_cartesian_descriptor(model)
# periodic = desc.isperiodic
# ncells = desc.partition

# inodes = LinearIndices(orders .* ncells .+ 1 .- periodic)

# cells = vec(CartesianIndices(ncells))
# tcells = view(cells,tface_to_mface)
# ordered_node_ids = TProduct.get_ordered_node_ids(inodes,diri_entities)

# k = DofsToODofs(fe_dof_basis,ordered_node_ids,orders)

# cache = return_cache(k,tcells[1],tface_to_mface[1])
# evaluate!(cache,k,tcells[1],tface_to_mface[1])

form(u,v) = ∫(u⋅v)dΩ
A = assemble_matrix(form,space,space)
A′ = assemble_matrix(form,space′,space′)

dmap = get_dof_map(model.model,space,get_dirichlet_entities(space′.spaces_1d))
A11 = A[dmap[:,:,1][:],dmap[:,:,1][:]]
A21 = A[dmap[:,:,2][:],dmap[:,:,1][:]]
A12 = A[dmap[:,:,1][:],dmap[:,:,2][:]]
A22 = A[dmap[:,:,2][:],dmap[:,:,2][:]]
Matrix([A11 A12
  A21 A22]) ≈ A′

# cellids = space.cell_dofs_ids
# cellids′ = Table(space′.cell_dof_ids)
# for cell in eachindex(cellids)
#   @assert A[cellids[cell]] ≈ A′[cellids′[cell]] "$cell"
# end

# using Gridap.Algebra

# a = SparseMatrixAssembler(space,space)
# matdata = collect_cell_matrix(space,space,∫(get_trial_fe_basis(space)⋅get_fe_basis(space))dΩ)
# m1 = nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
# symbolic_loop_matrix!(m1,a,matdata)
# m2 = nz_allocation(m1)
# # numeric_loop_matrix!(m2,a,matdata)
# rows_cache = array_cache(matdata[2][1])
# cols_cache = array_cache(matdata[3][1])
# vals_cache = array_cache(matdata[1][1])
# add! = AddEntriesMap(+)
# add_cache = nothing

# for cell in 1:length(matdata[3][1])
#   rows = getindex!(rows_cache,matdata[2][1],cell)
#   cols = getindex!(cols_cache,matdata[3][1],cell)
#   vals = getindex!(vals_cache,matdata[1][1],cell)
#   evaluate!(add_cache,add!,m2,vals,rows,cols)
# end

# cell = 2
# rows = getindex!(rows_cache,matdata[2][1],cell)
# cols = getindex!(cols_cache,matdata[3][1],cell)
# vals = getindex!(vals_cache,matdata[1][1],cell)
# # evaluate!(add_cache,add!,m2,vals,rows,cols)
# # add_entries!(+,m2,vals,rows,cols)
# lj,li = 1,1
# j,i = cols[lj],rows[li]
# vij = vals[li,lj]
# # add_entry!(+,m2,vij,i,j)

# # m3 = create_from_nz(m2)
# # m3

# a′ = SparseMatrixAssembler(space′,space′)
# matdata′ = collect_cell_matrix(space′,space′,∫(get_trial_fe_basis(space′)⋅get_fe_basis(space′))dΩ)
# m1′ = nz_counter(get_matrix_builder(a′),(get_rows(a′),get_cols(a′)))
# symbolic_loop_matrix!(m1′,a′,matdata′)
# m2′ = nz_allocation(m1′)
# # numeric_loop_matrix!(m2′,a′,matdata′)
# rows_cache′ = array_cache(matdata′[2][1])
# cols_cache′ = array_cache(matdata′[3][1])
# vals_cache′ = array_cache(matdata′[1][1])
# vals_sort = zeros(9,9)
# add_cache′ = nothing

# terms = DofMaps._get_terms(first(get_polytopes(bgmodel)),(2,2))
# terms = LinearIndices((3,3))[terms]
# invterms = invperm(terms)

# for cell in 1:length(matdata′[3][1])
#   rows′ = getindex!(rows_cache′,matdata′[2][1],cell)
#   cols′ = getindex!(cols_cache′,matdata′[3][1],cell)
#   vals′ = getindex!(vals_cache′,matdata′[1][1],cell)
#   for (i,J) in enumerate(terms)
#     for (j,I) in enumerate(terms)
#       # vals_sort[i,j] = vals′[I,J]
#       vals_sort[I,J] = vals′[i,j]
#     end
#   end
#   evaluate!(add_cache′,add!,m2′,vals_sort,rows′,cols′)
# end



# cell = 2
# rows′ = getindex!(rows_cache′,matdata′[2][1],cell)
# cols′ = getindex!(cols_cache′,matdata′[3][1],cell)
# vals′ = getindex!(vals_cache′,matdata′[1][1],cell)
# # evaluate!(add_cache′,add!,m2′,vals′,rows′,cols′)
# lj′,li′ = 1,1
# j′,i′ = cols′[lj′],rows[li′]
# vij′ = vals[li′,lj′]

# # m3′ = create_from_nz(m2′)
# # m3′

# linear system

j(u,v) = ∫(u⋅v)dΩ + ∫(∇(u)⊙∇(v))dΩ
l(v) = ∫(Point(1,1)⋅v)dΩ

op = AffineFEOperator(j,l,space,space)
op′ = AffineFEOperator(j,l,space′,space′)

uh = solve(op)
u = uh.free_values
uh′ = solve(op′)
u′ = uh′.free_values

u1 = u[dmap[:,:,1][:]]
u2 = u[dmap[:,:,2][:]]

vcat(u1,u2) ≈ u′

writevtk(Ω,"boh",cellfields=["err"=>uh-uh′])

# SPARSE MAP ?

A1 = assemble_matrix((u,v)->∫(u⋅v)*Measure(trian.trians_1d[1],2),space′.spaces_1d[1],space′.spaces_1d[1])
A2 = assemble_matrix((u,v)->∫(u⋅v)*Measure(trian.trians_1d[2],2),space′.spaces_1d[2],space′.spaces_1d[2])

using SparseArrays

tp_dof_map = get_tp_dof_map(space′)

I,J,V = findnz(A′)
i,j,v = univariate_findnz(osparsity)
sparse_indices = get_sparse_dof_map(osparsity,I,J,i,j)
osparse_indices = order_sparse_dof_map(osparsity,sparse_indices,rows′′,cols′′)
