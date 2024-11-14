using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.CellData

using ReducedOrderModels

n = 20
partition = (n,n)
model = CartesianDiscreteModel((0,1,0,1),partition)

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe,conformity=:H1;dirichlet_tags=["boundary"])

trian = Triangulation(model)

i = get_dof_map(model,test)
s = SparsityPattern(test,test)

trianv = view(trian,[1,10,100])
iv = change_domain(i,trianv)

cell_dof_ids = get_cell_dof_ids(test)


sparsity = SparsityPattern(trial,test,t...)
psparsity = order_sparsity(sparsity,trial,test)
I,J,_ = findnz(psparsity)
i,j,_ = DofMaps.univariate_findnz(psparsity)
g2l_sparse = _global_2_local(psparsity,I,J,i,j)
pg2l_sparse = _permute_dof_map(g2l_sparse,trial,test)
pg2l = to_nz_index(pg2l_sparse,sparsity)
SparseDofMap(pg2l,pg2l_sparse,psparsity)









































desc = get_cartesian_descriptor(model)

periodic = desc.isperiodic
ncells = desc.partition
ndofs = order .* ncells .+ 1 .- periodic

Dc,Ti = 2,Int32

terms = _get_terms(first(get_polytopes(model)),fill(order,Dc))
cache_cell_dof_ids = array_cache(cell_dof_ids)

ordered_dof_ids = LinearIndices(ndofs)
dof_map = zeros(Ti,ndofs)
dof_to_cell_ptrs = zeros(Ti,prod(ndofs)+1)
dof_to_cell_data = Ti[]
for (icell,cell) in enumerate(CartesianIndices(ncells))
  first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
  ordered_dofs_range = map(i -> i:i+order,first_new_dof)
  ordered_dofs = view(ordered_dof_ids,ordered_dofs_range...)
  cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
  for (idof,dof) in enumerate(cell_dofs)
    dof < 0 && continue
    t = terms[idof]
    odof = ordered_dofs[t]
    dof_map[odof] = dof
    dof_to_cell_ptrs[odof] += 1
    # dof_to_cell_ptrs[odof+1] += 1
    # push!(dof_to_cell_data,Ti(icell))
  end
end
length_to_ptrs!(dof_to_cell_ptrs)
ndata = dof_to_cell_ptrs[end]-1
dof_to_first_owner_cell = zeros(Ti,ndata)

# for ndof in 1:num_free_dofs(test)
#   ninds = findall(cell_dof_ids.data .== ndof)
#   nptrs = cell_dof_ids.ptrs[ninds]
#   dof_to_cell_data[nptrs] =
# end

for icell in 1:prod(ncells)

  cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
  for (idof,dof) in enumerate(cell_dofs)
    dof < 0 && continue
    dof_to_first_owner_cell[idof] = dof
  end
end
dof_to_cell = Table(dof_to_cell_data,dof_to_cell_ptrs)


cell_to_dof = cell_dof_ids
dof_to_cell, = make_inverse_table(cell_to_dof.data[findall(cell_to_dof.data .> 0)],num_free_dofs(test))

# DIO CANE

i2j = cell_to_dof.data[findall(cell_to_dof.data .> 0)]
nj = num_free_dofs(test)

ni = length(i2j)
@assert nj≥0

p = sortperm(i2j)
# i2j[p] is a sorted array of js

cache = array_cache(cell_dof_ids)

function return_cell(cache,cell_dof_ids,dof)
  cells = Int32[]
  for cell in 1:length(cell_dof_ids)
    cell_dofs = getindex!(cache,cell_dof_ids,cell)
    if dof ∈ cell_dofs
      append!(cells,cell)
    end
  end
  cells
end

@btime begin
  cells = lazy_map(dof -> return_cell($cache,$cell_dof_ids,dof),1:1521)
  Table(cells)
end

@btime begin
  cells = map(dof -> return_cell($cache,$cell_dof_ids,dof),1:1521)
  Table(cells)
end
