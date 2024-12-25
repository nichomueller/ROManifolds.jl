using Gridap
using Gridap.FESpaces
using Gridap.Algebra
using Gridap.Fields
using Gridap.Arrays
using GridapEmbedded
using GridapEmbedded.Interfaces
using ReducedOrderModels
using ReducedOrderModels.Utils
using DrWatson

u(x) = x[1] + x[2]#x[1]^2 + x[2]^2
f(x) = -Δ(u)(x)
ud(x) = u(x)

model = CartesianDiscreteModel((0,1,0,1),(4,4))

cells_in = collect(1:8)
cells_out = collect(9:16)
Ω = Triangulation(model)
Ω1 = view(Ω,cells_in)
Ω2 = view(Ω,cells_out)
Γd = BoundaryTriangulation(model;tags="boundary")
n_Γd = get_normal_vector(Γd)

dofs_in = collect(1:10)
dofs_Γ = collect(11:15)
dofs_out = collect(16:25)

order = 1
degree = 2*order

dΩ = Measure(Ω,degree)
dΩ1 = Measure(Ω1,degree)
dΩ2 = Measure(Ω2,degree)
dΓd = Measure(Γd,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

V = FESpace(Ω,reffe;conformity=:H1)
U = TrialFESpace(V)

γd = 10.0
γg = 0.1
h = 1/4

a1(u,v) = ∫(∇(v)⋅∇(u))*dΩ1
a2(u,v) = ∫(∇(v)⋅∇(u))*dΩ2
aΓ(u,v) = ∫( (γd/h)*v*u - v*(n_Γd⋅∇(u)) + (n_Γd⋅∇(v))*u ) * dΓd
a(u,v) = a1(u,v) + a2(u,v) + aΓ(u,v)

l1(v) = ∫(f*v)*dΩ1
l2(v) = ∫(f*v)*dΩ2
lΓ(v) = ∫( (γd/h)*v*ud + (n_Γd⋅∇(v))*ud ) * dΓd
l(v) = l1(v) + l2(v) + lΓ(v)

op = AffineFEOperator(a,l,U,V)
uh = solve(op)

writevtk(Ω,datadir("plts/sol"),cellfields=["uh"=>uh])

Vsplit = NewFESpace(V,Ω1)
Usplit = TrialFESpace(Vsplit)

v,du = get_fe_basis(V),get_trial_fe_basis(V)
veccontriball = ∫(∇(v)⋅∇(ud))*dΩ

struct MyMask <: Map end

function Arrays.return_cache(::MyMask,cellarray,cellids)
  z = zero(eltype(cellarray))
  a = similar(cellarray)
  return z,a
end

function Arrays.evaluate!(cache,::MyMask,cellarray,cellids)
  z,a = cache
  for (is,i) in enumerate(cellids)
    if i != 0
      a[is] = z
    else
      a[is] = cellarray[is]
    end
  end
  a
end

struct MyIdsMask <: Map end

function Arrays.return_cache(::MyIdsMask,cellids_with_zeros,cellids)
  z = zero(eltype(cellids_with_zeros))
  a = similar(cellids_with_zeros)
  return z,a
end

function Arrays.evaluate!(cache,::MyIdsMask,cellids_with_zeros,cellids)
  z,a = cache
  for (is,i) in enumerate(cellids_with_zeros)
    if i != 0
      a[is] = z
    else
      a[is] = cellids[is]
    end
  end
  a
end

veccontriboutin = lazy_map(MyMask(),veccontriball[Ω],Vsplit.cell_dof_ids)
idsoutin = lazy_map(MyIdsMask(),Vsplit.cell_dof_ids,get_cell_dof_ids(V))
vecdataoutin = ([veccontriboutin],[idsoutin])
assem = SparseMatrixAssembler(Usplit,Vsplit)

# A = assemble_matrix(a,Usplit,Vsplit)
# b = assemble_vector(l,Vsplit)
opsplit = AffineFEOperator(a,l,Usplit,Vsplit)
uhsplit = solve(opsplit)
A = opsplit.op.matrix; b = opsplit.op.vector
badd = assemble_vector(assem,vecdataoutin)

x = A \ b
x1 = A \ (b+badd)
x2 = A \ (b-badd)

uh1 = FEFunction(Usplit,x1)
uh2 = FEFunction(Usplit,x2)

writevtk(Ω,datadir("plts/sol_split"),cellfields=["uh1"=>uh1,"uh2"=>uh2])

@inline function Algebra._add_entries!(combine::Function,A,vs,is)
  println((is,vs))
  for (li, i) in enumerate(is)
    if i>0
      vi = vs[li]
      add_entry!(A,vi,i)
    end
  end
  A
end

A[dofs_in,dofs_in] == op.op.matrix[dofs_in,dofs_in]
A[dofs_in,dofs_Γ] == op.op.matrix[dofs_in,dofs_Γ]
A[dofs_Γ,dofs_in] == op.op.matrix[dofs_Γ,dofs_in]
A[dofs_in,dofs_out] == op.op.matrix[dofs_in,dofs_out] == A[dofs_out,dofs_in] == op.op.matrix[dofs_out,dofs_in]

A[dofs_Γ,dofs_Γ] == op.op.matrix[dofs_Γ,dofs_Γ]
A[dofs_out,dofs_Γ] == op.op.matrix[dofs_out,dofs_Γ]
A[dofs_Γ,dofs_out] == op.op.matrix[dofs_Γ,dofs_out]

b[dofs_in] == op.op.vector[dofs_in]
b[dofs_out] == op.op.vector[dofs_out]
b[dofs_Γ] == op.op.vector[dofs_Γ]

b[dofs_in] == op.op.vector[dofs_in]
x2[dofs_in] == uh.free_values[dofs_in]

x = get_cell_points(Ω)
duh = ∇(uh)
duhx = duh(x)

duh1 = ∇(uh1)
duh1x = duh1(x)

duh2 = ∇(uh2)
duh2x = duh2(x)

#

veccontribout = ∫(∇(v)⋅∇(ud))*dΩ_out
veccontriboutin = lazy_map(MyMask(),veccontribout[Ω_out],V.cell_dof_ids)
celldofidsout = get_cell_dof_ids(V,Ω_out)
_celldofidsout = get_cell_dof_ids(V.space,Ω_out)
idsoutin = lazy_map(MyIdsMask(),celldofidsout,_celldofidsout)
vecdataoutin = ([veccontriboutin],[idsoutin])
assem = SparseMatrixAssembler(U,V)

badd = assemble_vector(assem,vecdataoutin)
b′ = op.op.vector + badd
x′ = op.op.matrix \ b′
uh′ = FEFunction(U,x′)
writevtk(Ω_bg,datadir("plts/sol"),cellfields=["uh"=>uh])
