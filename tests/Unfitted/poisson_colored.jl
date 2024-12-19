using Gridap
using Gridap.FESpaces
using Gridap.Algebra
using GridapEmbedded
using GridapEmbedded.Interfaces
using ReducedOrderModels
using ReducedOrderModels.Utils

u(x) = x[1] - x[2]
f(x) = -Δ(u)(x)
ud(x) = u(x)

R = 0.5
L = 0.8*(2*R)
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(L,0.0)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin
const h = dp[1]/n

cutgeo = cut(bgmodel,geo3)

Ω_bg = Triangulation(bgmodel)
Ω_act = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT).b
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)

n_Γ = get_normal_vector(Γ)
n_Γg = get_normal_vector(Γg)

order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΩ_out = Measure(Ω_out,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

Vcut = FESpace(Ω_act,reffe)
Ucut = TrialFESpace(Vcut)

Vbg = FESpace(Ω_bg,reffe;conformity=:H1,dirichlet_tags="boundary")

# function _get_dofs(cellids::AbstractArray{<:AbstractArray})
#   _get_dofs(Table(cellids))
# end

# function _get_dofs(cellids::Table)
#   data = unique(cellids.data)
#   data[findall(data.>0)]
# end

# function _get_dof_to_active_inactive_tag(ndofs,active_dofs,inactive_dofs)
#   dof_to_tag = Vector{Int8}(undef,ndofs)
#   for i in 1:ndofs
#     dof_to_tag[i] = i ∈ active_dofs ? Int8(1) : Int8(2)
#   end
#   return dof_to_tag
# end

# function get_active_inactive_cells(Vbg::FESpace,Ωact::Triangulation)
#   Ωbg = get_triangulation(Vbg)
#   bg_cells = get_tface_to_mface(Ωbg)
#   @assert isa(bg_cells,IdentityVector)
#   active_cells = get_tface_to_mface(Ωact)
#   inactive_cells = setdiff(bg_cells,active_cells)
#   return (active_cells,inactive_cells)
# end

# function get_active_inactive_cellids(Vbg::FESpace,Ωact::Triangulation)
#   active_cells,inactive_cells = get_active_inactive_cells(Vbg,Ωact)
#   cellids = get_cell_dof_ids(Vbg)
#   active_cellids = lazy_map(Reindex(cellids),active_cells)
#   inactive_cellids = lazy_map(Reindex(cellids),inactive_cells)
#   return (active_cellids,inactive_cellids)
# end

# function active_inactive_dof_to_tag(Vbg::FESpace,Ωact::Triangulation)
#   active_cellids,inactive_cellids = get_active_inactive_cellids(Vbg,Ωact)
#   ndofs = num_free_dofs(Vbg)
#   active_dofs = _get_dofs(active_cellids)
#   inactive_dofs = setdiff(1:ndofs,active_dofs)
#   active_ndofs = length(active_dofs)
#   inactive_ndofs = length(inactive_dofs)
#   tags = ["active","inactive"]
#   tag_to_ndofs = [active_ndofs,inactive_ndofs]
#   dof_to_tag = _get_dof_to_active_inactive_tag(ndofs,active_dofs,inactive_dofs)
#   return tags,tag_to_ndofs,dof_to_tag
# end

# function get_interface_cellids(Vbg::FESpace,Ωout::Triangulation)
#   interface_cells = get_tface_to_mface(Ωout)
#   interface_cellids = get_cell_dof_ids(Vbg,Ωout)
#   return interface_cellids
# end

# function get_active_dofs_on_interface(Vbg::FESpace,Ωact::Triangulation,Ωout::Triangulation)
#   interface_cellids = get_interface_cellids(Vbg,Ωout)
#   active_cellids, = get_active_inactive_cellids(Vbg,Ωact)
#   active_dofs = _get_dofs(active_cellids)
#   return intersect(active_dofs,interface_dofs)
# end

# tags,tag_to_ndofs,dof_to_tag = active_inactive_dof_to_tag(Vbg,Ω_act)
# active_dofs_on_interface = get_active_dofs_on_interface(Vbg,Ω_act,Ω_out)
# dof_to_pdof = Utils.get_dof_to_colored_dof(tag_to_ndofs,dof_to_tag)

# Vcolor = ColoredFESpace(Vbg,Ω_act)
# Vcolor = MultiColorFESpace(Vbg,tags,tag_to_ndofs,dof_to_tag,dof_to_pdof)
# Ucolor = TrialFESpace(Vcolor)

γd = 10.0
γg = 0.1

acut(u,v) =
  ∫( ∇(v)⋅∇(u) ) * dΩ +
  ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u ) * dΓ +
  ∫( (γg*h)*jump(n_Γg⋅∇(v))*jump(n_Γg⋅∇(u)) ) * dΓg

aout(u,v) = ∫( ∇(v)⋅∇(u) ) * dΩ_out

a(u,v) = acut(u,v) + aout(u,v)

lcut(v) = ∫( v*f ) * dΩ + ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud ) * dΓ
lout(v) = ∫( ∇(v)⋅∇(ud) ) * dΩ_out
l(v) = lcut(v) - lout(v)

# using Gridap.FESpaces
# assem = SparseMatrixAssembler(Ucolor,Vcolor)
# du = get_trial_fe_basis(Ucolor)
# v = get_fe_basis(Vcolor)
# matcontribs,veccontribs = a(du,v),l(v)
# data = collect_cell_matrix_and_vector(Ucolor,Vcolor,matcontribs,veccontribs)
# A,b = assemble_matrix_and_vector(assem,data)
# x = zero_free_values(Vcolor)
# ldiv!(x,lu(A),b)

# op = AffineFEOperator(acut,l,Ucut,Vcut)
# uh = solve(op)

# sqrt(norm(x[Block(1)])^2 + norm(x[Block(2)])^2) ≈ norm(uh.free_values)

# assem_cut = SparseMatrixAssembler(Ucut,Vcut)
# matcontribscut,veccontribscut = acut(du,v),veccontribs
# data_cut = collect_cell_matrix_and_vector(Ucut,Vcut,matcontribscut,veccontribscut)
# A_cut,b_cut = assemble_matrix_and_vector(assem_cut,data_cut)

# cellids_cut = get_cell_dof_ids(Vcut)
# cellids_color = get_cell_dof_ids(Vcolor)

# tface_act = get_tface_to_mface(Ω_act)
# for (cell_act,cell_color) in enumerate(tface_act)
#   ids_cut = cellids_cut[cell_act]
#   ids_color = cellids_color[cell_color]
#   @assert length(ids_cut)==length(ids_color)
#   for i in 1:length(ids_cut)
#     colori = ids_color.colors[][i]
#     @assert b_cut[ids_cut[i]] ≈ b[Block(colori)][ids_color.array[][i]]
#   end
# end

# for (cell_act,cell_color) in enumerate(tface_act)
#   ids_cut = cellids_cut[cell_act]
#   ids_color = cellids_color[cell_color]
#   @assert length(ids_cut)==length(ids_color)
#   for i in 1:length(ids_cut)
#     colori = ids_color.colors[][i]
#     arrayi = ids_color.array[][i]
#     ids_cuti = ids_cut[i]
#     for j in 1:length(ids_cut)
#       colorj = ids_color.colors[][j]
#       arrayj = ids_color.array[][j]
#       ids_cutj = ids_cut[j]
#       @assert A_cut[ids_cuti,ids_cutj] ≈ A[Block(colori,colorj)][arrayi,arrayj] "$cell_act"
#     end
#   end
# end

Vnew = NewFESpace(Vbg,Ω_act)
Unew = TrialFESpace(Vnew,ud)
opnew = AffineFEOperator(a,l,Unew,Vnew)
uhnew = solve(opnew)

# ids = -unique(Vnew.cell_dof_ids.data[findall(Vnew.cell_dof_ids.data .< 0)])
bnew = opnew.op.vector
# bnewids = bnew[ids]

tface_in = get_tface_to_mface(Ω_act)
tface_out = get_tface_to_mface(Ω_out)
cell = 1
for cell_out in tface_out
  ids = Vnew.cell_dof_ids[cell_out]
  if any(ids.<0)
    cell = cell_out
    break
  end
end
ids = abs.(Vnew.cell_dof_ids[cell])
bnew[ids]

v = get_fe_basis(Vnew)
assem = SparseMatrixAssembler(Vnew,Vnew)
vecdata = collect_cell_vector(Vnew,l(v))

vd = (vecdata[1][2],vecdata[2][2])
v1 = nz_counter(get_vector_builder(assem),(get_rows(assem),))
symbolic_loop_vector!(v1,assem,vd)
v2 = nz_allocation(v1)
numeric_loop_vector!(v2,assem,vd)
v3 = create_from_nz(v2)

writevtk(Ω_bg,datadir("plts/sol"),cellfields=["uh"=>uhnew])

using Gridap.Algebra

x = opnew.op.matrix \ opnew.op.vector

b̃ = opnew.op.vector - boutok
x̃ = opnew.op.matrix \ b̃
x̃h = FEFunction(Unew,x̃)
writevtk(Ω_bg,datadir("plts/sol_new"),cellfields=["uh"=>x̃h])
