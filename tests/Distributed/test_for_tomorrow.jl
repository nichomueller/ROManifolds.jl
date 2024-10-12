using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.FESpaces
using ReducedOrderModels.FEM
using ReducedOrderModels.Distributed
using GridapDistributed
using PartitionedArrays
using LinearAlgebra
using SparseArrays
using Test

parts = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(parts),)))
end

domain = (0,4,0,4)
cells = (4,4)
model = CartesianDiscreteModel(ranks,parts,domain,cells)
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

# pranges = fill([0,1],3)
# pspace = ParamSpace(pranges)
# μ = Realization([[1.0],[1.0],[1.0]])

# f(x,μ) = sum(μ)
# f(μ) = x->f(x,μ)
# fμ = ParamFunction(f,μ)
# u(x,μ) = (x[1]+x[2])*sum(μ)
# u(μ) = x->u(x,μ)
# uμ = ParamFunction(u,μ)
# reffe = ReferenceFE(lagrangian,Float64,1)
# V0 = TestFESpace(model,reffe,dirichlet_tags="boundary")
# U = TrialParamFESpace(uμ,V0)

# a(u,v) = ∫(fμ*∇(v)⋅∇(u))dΩ
# l(v) = ∫(fμ*v)dΩ

f(x) = 1
u(x) = x[1]+x[2]
reffe = ReferenceFE(lagrangian,Float64,1)
V0 = TestFESpace(model,reffe,dirichlet_tags="boundary")
U = TrialFESpace(u,V0)

a(u,v) = ∫(∇(v)⋅∇(u))dΩ
l(v) = ∫(v)dΩ

dv = get_fe_basis(V0)
du = get_trial_fe_basis(U)
assem = SparseMatrixAssembler(U,V0,SubAssembledRows())
zh = zero(U)
data = collect_cell_matrix_and_vector(U,V0,a(du,dv),l(dv),zh)
A,b = assemble_matrix_and_vector(assem,data)

x = A \ b

# ################################ define basis ##################################
# row_partition = A.row_partition
# own_vals = map(own_values(A)) do Aloc
#   Float64.(I(size(Aloc,1)))
# end
# col_partition = Distributed.get_col_partition(own_vals,row_partition)
# # col_partition = V0.gids.partition
# Φ = PMatrix{Matrix{Float64}}(undef,row_partition,col_partition)
# map(own_values(Φ),own_vals) do Φo,vo
#   Φo .= vo
# end
# consistent!(Φ) |> wait
# ################################ define basis ##################################

# ################################## products ####################################

# # matrix reduction
# Φ_compatible = GridapDistributed.change_ghost(Φ,axes(A,2);make_consistent=true)
# ΦᵀAΦ_oo = map(own_values(Φ),own_values(A),own_values(Φ_compatible)) do Φo,Aoo,Φco
#   Φo'*Aoo*Φco
# end
# ΦᵀAΦ_og = map(own_values(Φ),own_ghost_values(A),ghost_values(Φ_compatible)) do Φo,Aog,Φcg
#   Φo'*Aog*Φcg
# end
# ΦᵀAΦ_go = map(ghost_values(Φ),ghost_values(A),own_values(Φ_compatible)) do Φg,Ago,Φco
#   Φg'*Ago*Φco
# end
# ΦᵀAΦ_gg = map(ghost_values(Φ),ghost_own_values(A),ghost_values(Φ_compatible)) do Φg,Agg,Φcg
#   Φg'*Agg*Φcg
# end

# # vector reduction
# c = PVector{Vector{Float64}}(undef,partition(axes(b,1)))
# t = consistent!(b) |> wait
# map(own_values(c),own_values(Φ),own_values(b)) do co,Φo,bo
#   mul!(co,Φo',bo,1,1)
# end
# consistent!(c) |> wait
# @test norm(b-c) ≈ 0.0

################################################################################
function my_id(n)
  map(1:n) do i
    v = zeros(Float64,n)
    v[i] = 1.0
    v
  end
end
row_partition = A.row_partition
loc_vals = map(local_values(A)) do Aloc
  iv = my_id(size(Aloc,1))
  ParamArray(iv)
end
Φ = PVector(loc_vals,row_partition)
# consistent!(Φ) |> wait
################################################################################
Φc = GridapDistributed.change_ghost(Φ,axes(A,2);make_consistent=true)
A*Φc
Φ'*A
ΦAΦ = Φ'*A*Φc

Φb = Φ'*b

x_reduced = map_main(ΦAΦ,Φb) do A,b
  A \ b
end
x_reduced1 = x_reduced.items[1]

x_recast = copy(x)
fill!(x_recast,0.0)
map(own_values(x_recast),own_values(Φ)) do xfull,Φ
  xfull .= hcat(Φ.array...)*x_reduced1
end
consistent!(x_recast) |> wait
consistent!(x) |> wait
norm(x_recast - x)

xtrue = PartitionedArrays.to_trivial_partition(x,partition(axes(A_trivial,1)))
xtrue1 = xtrue.vector_partition.items[1]

aa = Φ'
row_partition_in_main = PartitionedArrays.trivial_partition(partition(axes(b,1)))
a_in_main = PartitionedArrays.to_trivial_partition(aa.parent,row_partition_in_main)
b_in_main = PartitionedArrays.to_trivial_partition(b,row_partition_in_main)

mya = a_in_main.vector_partition.items[1]
myb = b_in_main.vector_partition.items[1]
c = zeros(length(mya),length(myb))

A_trivial = PartitionedArrays.to_trivial_partition(A)
A_main = A_trivial.matrix_partition.items[1]

Φ_trivial = PartitionedArrays.to_trivial_partition(Φ,partition(axes(A_trivial,1)))
Φ_main = Φ_trivial.vector_partition.items[1]

Φ_matrix = hcat(Φ_main.array...)
ΦAΦ = Φ_matrix'*A_main*Φ_matrix

PP = Φ'*Φ

############# more tests
function Base.setindex!(a::PartitionedArrays.SubSparseMatrix,v,i::Integer,j::Integer)
  I = a.indices[1][i]
  J = a.indices[2][j]
  a.parent[I,J] = v
end

x_approx = copy(x)
fill!(x_approx,0.0)
map(own_values(x_approx),own_values(Φ),own_values(x)) do xapp,Φ,x
  basis = hcat(Φ...)
  xapp .= basis*basis'*x
end
consistent!(x_approx) |> wait
norm(x - x_approx)

@boundscheck @assert PartitionedArrays.matching_own_indices(axes(x,1),axes(b,1))
b_approx = copy(b)
fill!(b_approx,0.0)
map(own_values(b_approx),own_values(Φ),own_values(b)) do bapp,Φ,b
  basis = hcat(Φ...)
  bapp .= basis*basis'*b
end
norm(b - b_approx)

Φc = GridapDistributed.change_ghost(Φ,axes(A,2);make_consistent=true)
Arc = A*Φc
@boundscheck @assert PartitionedArrays.matching_own_indices(axes(x,1),axes(Arc,1))
Arc_approx = copy(Arc)
fill!(Arc_approx,0.0)
map(own_values(Arc_approx),own_values(Φ),own_values(Arc)) do Arcapp,Φ,Arc
  basis = hcat(Φ...)
  Arcapp .= ParamArray([basis*basis'*Arc[i] for i = eachindex(Arc)])
end
norm(Arc - Arc_approx)

A_approx = copy(A)
LinearAlgebra.fillstored!(A_approx,0.0)
map(own_values(A_approx),own_values(Φ),own_values(Φc),own_values(A)) do Aapp,Φ,Φc,A
  basis = hcat(Φ...)
  basisc = hcat(Φc...)
  Ared = basis'*A*basisc
  Aapp .= basis*Ared*basisc'
end
map(own_ghost_values(A_approx),own_values(Φ),ghost_values(Φc),own_ghost_values(A)) do Aapp,Φ,Φc,A
  basis = hcat(Φ...)
  basisc = hcat(Φc...)
  Ared = basis'*A*basisc
  Aapp .= basis*Ared*basisc'
end
err = map(own_values(A),own_values(A_approx)) do A,Aapp
  norm(A-Aapp)
end

coeff_b = map(own_values(Φ),own_values(b)) do Φ,b
  basis = hcat(Φ...)
  basis'*b
end

coeff_A = map(own_values(Φ),own_values(A),own_values(Φc),own_ghost_values(A),ghost_values(Φc)) do Φ,A,Φc,Ag,Φgc
  basis = hcat(Φ...)
  basisc = hcat(Φc...)
  basisgc = hcat(Φgc...)
  basis'*(A*basisc + Ag*basisgc)
end

coeff_x = map(coeff_A,coeff_b) do A,b
  nnz_ids = findall(!iszero,b)
  x = zeros(size(b))
  x[nnz_ids] .= A[nnz_ids,nnz_ids] \ b[nnz_ids]
  x
end

x_approx = copy(x)
fill!(x_approx,0.0)
map(own_values(x_approx),own_values(Φ),coeff_x) do xapp,Φ,xred
  basis = hcat(Φ...)
  xapp .= basis*xred
end
norm(x - x_approx)


# map(own_values(Φ),own_values(A),own_values(Φc),own_ghost_values(A),ghost_values(Φc)) do Φ,A,Φc,Ag,Φgc

basis = hcat(own_values(Φ).items[1]...)
basisc = hcat(own_values(Φc).items[1]...)
basisgc = hcat(ghost_values(Φc).items[1]...)
Ao = own_values(A).items[1]
Aog = own_ghost_values(A).items[1]
vo = basis'*Ao*basisc
vog = basis'*Aog*basisgc
_Ao = basis*vo*basisc'
_Aog = basis*vog*basisgc'


_coeff_A = map(local_values(Φ),local_values(A),local_values(Φc)) do Φ,A,Φc
  basis = hcat(Φ...)
  basisc = hcat(Φc...)
  basis'*A*basisc
end
