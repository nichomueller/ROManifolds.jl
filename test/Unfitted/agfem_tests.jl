using Gridap
using GridapEmbedded
using ROManifolds
using Test
using Gridap.FESpaces
using Gridap.Arrays

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,5])
add_tag_from_tags!(labels,"neumann",[6,7,8])

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model,tags="neumann")
dΩ = Measure(Ω,2)
dΓ = Measure(Γ,2)

V = FESpace(
  model,ReferenceFE(lagrangian,Float64,2), conformity=:H1) #, dirichlet_tags="dirichlet"

Vo = OrderedFESpace(model,ReferenceFE(lagrangian,Float64,2),conformity=:H1) #,dirichlet_tags="dirichlet"

# fdof_to_val = collect(Float64,1:num_free_dofs(V))
# ddof_to_val = -collect(Float64,1:num_dirichlet_dofs(V))
# vh = FEFunction(V,fdof_to_val,ddof_to_val)

# sDOF_to_dof = [1,5,-2]
# sDOF_to_dofs = Table([[-1,4],[4,6],[-1,-3]])
# sDOF_to_coeffs = Table([[0.5,0.5],[0.5,0.5],[0.5,0.5]])
sDOF_to_dof = [2,6]
sDOF_to_dofs = Table([[1,3],[3,9]])
sDOF_to_coeffs = Table([[0.5,0.5],[0.5,0.5]])

Vc = FESpaceWithLinearConstraints(
  sDOF_to_dof,
  sDOF_to_dofs,
  sDOF_to_coeffs,
  V)

sDOF_to_odof = [3,15]
sDOF_to_odofs = Table([[1,5],[25,5]])

VOc = FESpaceWithLinearConstraints(
  sDOF_to_odof,
  sDOF_to_odofs,
  sDOF_to_coeffs,
  Vo)

u(x) = x[1] + 2*x[2]
f(x) = -Δ(u)(x)

Uc = TrialFESpace(Vc,u)
@test has_constraints(Uc)
uch = interpolate(u,Uc)

n_Γ = get_normal_vector(Γ)

a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ #+ ∫( jump(u)*jump(v) )*dΛ
l(v) = ∫( v*f )*dΩ + ∫( v*(n_Γ⋅∇(u)) )*dΓ

op = AffineFEOperator(a,l,Uc,Vc)
uch = solve(op)

# #using Gridap.Visualization
# #writevtk(trian,"trian",nsubcells=10,cellfields=["uch"=>uch])

e = u - uch

e_l2 = sqrt(sum(∫( e*e )*dΩ))
e_h1 = sqrt(sum(∫( e*e + ∇(e)⋅∇(e) )*dΩ))

UOc = TrialFESpace(VOc,u)
@test has_constraints(UOc)
ucho = interpolate(u,UOc)

a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ #+ ∫( jump(u)*jump(v) )*dΛ
l(v) = ∫( v*f )*dΩ + ∫( v*(n_Γ⋅∇(u)) )*dΓ

opo = AffineFEOperator(a,l,UOc,VOc)
ucho = solve(opo)

# #using Gridap.Visualization
# #writevtk(trian,"trian",nsubcells=10,cellfields=["uch"=>uch])

eo = u - ucho

eo_l2 = sqrt(sum(∫( eo*eo )*dΩ))
eo_h1 = sqrt(sum(∫( eo*eo + ∇(eo)⋅∇(eo) )*dΩ))

# tol = 1.e-9
# @test e_l2 < tol
# @test e_h1 < tol

# space = V
# n_fdofs = num_free_dofs(space)
# n_ddofs = num_dirichlet_dofs(space)
# n_DOFs = n_fdofs + n_ddofs

# DOF_to_DOFs, DOF_to_coeffs = FESpaces._prepare_DOF_to_DOFs(
#   sDOF_to_dof, sDOF_to_dofs, sDOF_to_coeffs, n_fdofs, n_DOFs)

# n_fdofs = num_free_dofs(space)
# mDOF_to_DOF, n_fmdofs = FESpaces._find_master_dofs(DOF_to_DOFs,n_fdofs)
# DOF_to_mDOFs = FESpaces._renumber_constraints!(DOF_to_DOFs,mDOF_to_DOF)
# cell_to_ldof_to_dof = Table(get_cell_dof_ids(space))
# cell_to_lmdof_to_mdof = __setup_cell_to_lmdof_to_mdof(cell_to_ldof_to_dof,DOF_to_mDOFs,n_fdofs,n_fmdofs)


# ospace = Vo
# no_fdofs = num_free_dofs(ospace)
# no_ddofs = num_dirichlet_dofs(ospace)
# no_DOFs = no_fdofs + no_ddofs

# DOF_to_ODOFs, DOF_to_ocoeffs = FESpaces._prepare_DOF_to_DOFs(
#   sDOF_to_odof, sDOF_to_odofs, sDOF_to_coeffs, no_fdofs, no_DOFs)

# no_fdofs = num_free_dofs(ospace)
# mDOF_to_ODOF,no_fmdofs = FESpaces._find_master_dofs(DOF_to_ODOFs,no_fdofs)
# DOF_to_mODOFs = FESpaces._renumber_constraints!(DOF_to_ODOFs,mDOF_to_ODOF)
# cell_to_lodof_to_odof = Table(get_cell_dof_ids(ospace))
# cell_to_lomdof_to_modof = FESpaces._setup_cell_to_lmdof_to_mdof(cell_to_lodof_to_odof,DOF_to_mODOFs,no_fdofs,no_fmdofs)


# function __setup_cell_to_lmdof_to_mdof(cell_to_ldof_to_dof,DOF_to_mDOFs,n_fdofs,n_fmdofs)

#   n_cells = length(cell_to_ldof_to_dof)
#   cell_to_lmdof_to_mdof_ptrs = zeros(eltype(cell_to_ldof_to_dof.ptrs),n_cells+1)

#   for cell in 1:n_cells
#     mdofs = Set{Int}()
#     pini = cell_to_ldof_to_dof.ptrs[cell]
#     pend = cell_to_ldof_to_dof.ptrs[cell+1]-1
#     for p in pini:pend
#       dof = cell_to_ldof_to_dof.data[p]
#       DOF = _dof_to_DOF(dof,n_fdofs)
#       qini = DOF_to_mDOFs.ptrs[DOF]
#       qend = DOF_to_mDOFs.ptrs[DOF+1]-1
#       for q in qini:qend
#         mDOF = DOF_to_mDOFs.data[q]
#         mdof = _DOF_to_dof(mDOF,n_fmdofs)
#         push!(mdofs,mdof)
#       end
#     end
#     cell_to_lmdof_to_mdof_ptrs[cell+1] = length(mdofs)
#   end

#   length_to_ptrs!(cell_to_lmdof_to_mdof_ptrs)
#   ndata = cell_to_lmdof_to_mdof_ptrs[end]-1
#   cell_to_lmdof_to_mdof_data = zeros(eltype(cell_to_ldof_to_dof.data),ndata)

#   for cell in 1:n_cells
#     mdofs = Set{Int}()
#     pini = cell_to_ldof_to_dof.ptrs[cell]
#     pend = cell_to_ldof_to_dof.ptrs[cell+1]-1
#     for p in pini:pend
#       dof = cell_to_ldof_to_dof.data[p]
#       DOF = _dof_to_DOF(dof,n_fdofs)
#       qini = DOF_to_mDOFs.ptrs[DOF]
#       qend = DOF_to_mDOFs.ptrs[DOF+1]-1
#       for q in qini:qend
#         mDOF = DOF_to_mDOFs.data[q]
#         mdof = _DOF_to_dof(mDOF,n_fmdofs)
#         println(mdof)
#         push!(mdofs,mdof)
#       end
#     end

#     o = cell_to_lmdof_to_mdof_ptrs[cell]-1
#     for (lmdof, mdof) in enumerate(mdofs)
#       cell_to_lmdof_to_mdof_data[o+lmdof] = mdof
#     end
#   end

#   Table(cell_to_lmdof_to_mdof_data,cell_to_lmdof_to_mdof_ptrs)
# end

#

geo1 = square(L=1,x0=Point(0,0))
geo2 = square(L=1,x0=Point(0.5,0))
geo = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R
dp = pmax - pmin

n=20
partition = (n,n)

model = CartesianDiscreteModel(pmin,pmax,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"neumann",[6,7,8])

cutgeo = cut(model,geo)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model,tags="neumann")
dΩ = Measure(Ω,2)
dΓ = Measure(Γ,2)

Ω_act = Triangulation(cutgeo,ACTIVE)

V = FESpace(Ω_act,ReferenceFE(lagrangian,Float64,2),conformity=:H1)
Vo = OrderedFESpace(Ω_act,ReferenceFE(lagrangian,Float64,2),conformity=:H1)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Vc = AgFEMSpace(V,aggregates)
Voc = AgFEMSpace(Vo,aggregates)
