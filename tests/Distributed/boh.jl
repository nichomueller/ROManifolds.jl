m1 = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
v1 = nz_counter(get_vector_builder(assem),(get_rows(assem),))
symbolic_loop_matrix_and_vector!(m1,v1,assem,data)
m2,v2 = nz_allocation(m1,v1)
numeric_loop_matrix_and_vector!(m2,v2,assem,data)
create_from_nz(m2,v2)

_U = TrialFESpace(u(rand(3)),V0)
_assem = SparseMatrixAssembler(_U,V0,das)
_data = collect_cell_matrix_and_vector(_U,V0,_a(du,dv),_l(dv),zero(_U))
# A1,b1 = assemble_matrix_and_vector(_assem,_data)
  _m1 = nz_counter(get_matrix_builder(_assem),(get_rows(_assem),get_cols(_assem)))
  _v1 = nz_counter(get_vector_builder(_assem),(get_rows(_assem),))
  symbolic_loop_matrix_and_vector!(_m1,_v1,_assem,_data)
  _m2,_v2 = nz_allocation(_m1,_v1)
  numeric_loop_matrix_and_vector!(_m2,_v2,_assem,_data)
  create_from_nz(_m2,_v2)
#############################################################

function my_check(a,b;i=1)
  msg = "Failed at $i"
  map(local_views(a),local_views(b)) do a,b
    @assert a.array[i].I == b.I msg
    @assert a.array[i].J == b.J msg
    # @assert a.array[i].V == b.V msg
  end
end

function check_on_allocation_coo(U,V0,a,_a,du,dv,assem,μ)
  matdata = collect_cell_matrix(U,V0,a(du,dv))
  m1 = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
  symbolic_loop_matrix!(m1,assem,matdata)
  m2 = nz_allocation(m1)
  numeric_loop_matrix!(m2,assem,matdata)

  for (i,μi) in enumerate(FEM._get_params(μ))
    _U = TrialFESpace(u(μi),V0)
    _assem = SparseMatrixAssembler(_U,V0,das)
    _matdata = collect_cell_matrix(_U,V0,_a(μi,du,dv))
    _m1 = nz_counter(get_matrix_builder(_assem),(get_rows(_assem),get_cols(_assem)))
    symbolic_loop_matrix!(_m1,_assem,_matdata)
    _m2 = nz_allocation(_m1)
    numeric_loop_matrix!(_m2,_assem,_matdata)
    my_check(m2,_m2;i)
  end
end

function check_on_coo_output(U,V0,a,_a,du,dv,assem,μ)
  matdata = collect_cell_matrix(U,V0,a(du,dv))
  m1 = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
  symbolic_loop_matrix!(m1,assem,matdata)
  m2 = nz_allocation(m1)
  numeric_loop_matrix!(m2,assem,matdata)

  I,J,V = GridapDistributed.get_allocations(m2)
  test_dofs_gids_prange = GridapDistributed.get_test_gids(m2)
  trial_dofs_gids_prange = GridapDistributed.get_trial_gids(m2)
  GridapDistributed.to_global_indices!(I,test_dofs_gids_prange;ax=:rows)
  GridapDistributed.to_global_indices!(J,trial_dofs_gids_prange;ax=:cols)
  rows = GridapDistributed._setup_prange(test_dofs_gids_prange,I;ax=:rows)
  Jo = GridapDistributed.get_gid_owners(J,trial_dofs_gids_prange;ax=:cols)
  t  = GridapDistributed._assemble_coo!(I,J,V,rows;owners=Jo)

  wait(t)
  cols = GridapDistributed._setup_prange(trial_dofs_gids_prange,J;ax=:cols,owners=Jo)

  GridapDistributed.to_local_indices!(I,rows;ax=:rows)
  GridapDistributed.to_local_indices!(J,cols;ax=:cols)
  asys = GridapDistributed.change_axes(m2,(rows,cols))

  for (i,μi) in enumerate(FEM._get_params(μ))
    _U = TrialFESpace(u(μi),V0)
    _assem = SparseMatrixAssembler(_U,V0,das)
    _matdata = collect_cell_matrix(_U,V0,_a(μi,du,dv))
    _m1 = nz_counter(get_matrix_builder(_assem),(get_rows(_assem),get_cols(_assem)))
    symbolic_loop_matrix!(_m1,_assem,_matdata)
    _m2 = nz_allocation(_m1)
    numeric_loop_matrix!(_m2,_assem,_matdata)

    _I,_J,_V = GridapDistributed.get_allocations(_m2)
    _test_dofs_gids_prange = GridapDistributed.get_test_gids(_m2)
    _trial_dofs_gids_prange = GridapDistributed.get_trial_gids(_m2)
    GridapDistributed.to_global_indices!(_I,_test_dofs_gids_prange;ax=:rows)
    GridapDistributed.to_global_indices!(_J,_trial_dofs_gids_prange;ax=:cols)
    _rows = GridapDistributed._setup_prange(_test_dofs_gids_prange,_I;ax=:rows)
    _Jo = GridapDistributed.get_gid_owners(_J,_trial_dofs_gids_prange;ax=:cols)
    _t  = GridapDistributed._assemble_coo!(_I,_J,_V,_rows;owners=_Jo)

    wait(_t)
    _cols = GridapDistributed._setup_prange(_trial_dofs_gids_prange,_J;ax=:cols,owners=_Jo)

    GridapDistributed.to_local_indices!(_I,_rows;ax=:rows)
    GridapDistributed.to_local_indices!(_J,_cols;ax=:cols)
    _asys = GridapDistributed.change_axes(_m2,(_rows,_cols))

    my_check(m2,_m2;i=1)
  end
end

check_on_allocation_coo(U,V0,a,_a,du,dv,assem,μ)
check_on_coo_output(U,V0,a,_a,du,dv,assem,μ)



# alternative 1
function my_get_allocations(a::Distributed.DistributedParamAllocationCOO)
  I,J,V = map(local_views(a)) do alloc
    i,j,v = map(alloc) do a
      a.I,a.J,a.V
    end |> tuple_of_arrays
    ParamContainer(i),ParamContainer(j),ParamContainer(v)
  end |> tuple_of_arrays
  return I,J,V
end

function my_to_local!(I,indices)
  map(I) do I
    PartitionedArrays.to_local!(I,indices)
  end
end
function my_to_global!(I,indices)
  map(I) do I
    PartitionedArrays.to_global!(I,indices)
  end
end

function my_to_local_indices!(I,ids::PRange;kwargs...)
  map(my_to_local!,I,partition(ids))
end

function my_to_global_indices!(I,ids::PRange;kwargs...)
  map(my_to_global!,I,partition(ids))
end

matdata = collect_cell_matrix(U,V0,a(du,dv))
ϵ = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
symbolic_loop_matrix!(ϵ,assem,matdata)
δ = nz_allocation(ϵ)
numeric_loop_matrix!(δ,assem,matdata)
# α,β,γ = my_get_allocations(δ)
# my_to_global_indices!(α,test_dofs_gids_prange;ax=:rows)
# my_to_global_indices!(β,trial_dofs_gids_prange;ax=:cols)
α,β,γ = GridapDistributed.get_allocations(δ)
GridapDistributed.to_global_indices!(α,test_dofs_gids_prange;ax=:rows)
GridapDistributed.to_global_indices!(β,trial_dofs_gids_prange;ax=:cols)

_U = TrialFESpace(u(μi),V0)
_assem = SparseMatrixAssembler(_U,V0,das)
_matdata = collect_cell_matrix(_U,V0,_a(μi,du,dv))
_m1 = nz_counter(get_matrix_builder(_assem),(get_rows(_assem),get_cols(_assem)))
symbolic_loop_matrix!(_m1,_assem,_matdata)
_m2 = nz_allocation(_m1)
numeric_loop_matrix!(_m2,_assem,_matdata)
_I,_J,_V = GridapDistributed.get_allocations(_m2)
GridapDistributed.to_global_indices!(_I,_test_dofs_gids_prange;ax=:rows)
GridapDistributed.to_global_indices!(_J,_trial_dofs_gids_prange;ax=:cols)

map(local_views(δ),local_views(_m2)) do a,b
  @assert a.array[3].I == b.I
  @assert a.array[3].J == b.J
end

# alternative 2 doesnt work
function my_get_allocations(a::Distributed.DistributedParamAllocationCOO)
  I,J,V = map(local_views(a)) do alloc
    i = ParamContainer(fill(first(alloc).I,length(alloc)))
    j = ParamContainer(fill(first(alloc).J,length(alloc)))
    v = ParamContainer(map(a->a.V,alloc))
    i,j,v
  end |> tuple_of_arrays
  return I,J,V
end

function my_to_local!(I::AbstractParamContainer,indices)
  PartitionedArrays.to_local!(first(I),indices)
end
function my_to_global!(I::AbstractParamContainer,indices)
  PartitionedArrays.to_global!(first(I),indices)
end
function my_to_local_indices!(I,ids::PRange;kwargs...)
  map(my_to_local!,I,partition(ids))
end
function my_to_global_indices!(I,ids::PRange;kwargs...)
  map(my_to_global!,I,partition(ids))
end

matdata = collect_cell_matrix(U,V0,a(du,dv))
ϵ = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
symbolic_loop_matrix!(ϵ,assem,matdata)
δ = nz_allocation(ϵ)
numeric_loop_matrix!(δ,assem,matdata)
# α,β,γ = my_get_allocations(δ)
# my_to_global_indices!(α,test_dofs_gids_prange;ax=:rows)
# my_to_global_indices!(β,trial_dofs_gids_prange;ax=:cols)
α,β,γ = my_get_allocations(δ)
my_to_global_indices!(α,test_dofs_gids_prange;ax=:rows)
my_to_global_indices!(β,trial_dofs_gids_prange;ax=:cols)

_U = TrialFESpace(u(μi),V0)
_assem = SparseMatrixAssembler(_U,V0,das)
_matdata = collect_cell_matrix(_U,V0,_a(μi,du,dv))
_m1 = nz_counter(get_matrix_builder(_assem),(get_rows(_assem),get_cols(_assem)))
symbolic_loop_matrix!(_m1,_assem,_matdata)
_m2 = nz_allocation(_m1)
numeric_loop_matrix!(_m2,_assem,_matdata)
_I,_J,_V = GridapDistributed.get_allocations(_m2)
GridapDistributed.to_global_indices!(_I,_test_dofs_gids_prange;ax=:rows)
GridapDistributed.to_global_indices!(_J,_trial_dofs_gids_prange;ax=:cols)

map(local_views(δ),local_views(_m2)) do a,b
  @assert a.array[3].I == b.I
  @assert a.array[3].J == b.J
end









datamat = collect_cell_matrix(U,V,a(du,dv))
datamat1 = collect_cell_matrix(U1,V,a11(du,dv))

datavec = collect_cell_vector(V,l(dv))
datavec1 = collect_cell_vector(V,l1(dv))

i = 1
map(local_views(datamat),local_views(datamat1)) do d,d1
  dm,dr,dc = d
  dm1,dr1,dc1 = d1
  @assert lazy_map(x->getindex(x,i),dm[1]) ≈ dm1[1]*sum(FEM._get_params(μ)[i])
  @assert dr == dr1
  @assert dc == dc1
end
map(local_views(datavec),local_views(datavec1)) do d,d1
  dv, = d
  dv1, = d1
  @assert lazy_map(x->getindex(x,1),dv[1]) ≈ dv1[1]*sum(μ1)
end



function GridapDistributed.assemble_coo_with_column_owner!(I,J,V,row_partition,Jown)
  """
    Returns three JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_entries_with_column_owner)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi, k_gj, k_jo, k_v = coo_entries_with_column_owner
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    PartitionedArrays.length_to_ptrs!(ptrs)
    gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
    gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
    jo_snd_data = zeros(eltype(k_jo),ptrs[end]-1)
    v_snd_data = zeros(eltype(k_v),ptrs[end]-1)
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        gj = k_gj[k]
        v = k_v[k]
        p = ptrs[owner_to_i[owner]]
        gi_snd_data[p] = gi
        gj_snd_data[p] = gj
        jo_snd_data[p] = k_jo[k]
        v_snd_data[p]  = v
        k_v[k] = zero(v)
        ptrs[owner_to_i[owner]] += 1
      end
    end
    PartitionedArrays.rewind_ptrs!(ptrs)
    gi_snd = JaggedArray(gi_snd_data,ptrs)
    gj_snd = JaggedArray(gj_snd_data,ptrs)
    jo_snd = JaggedArray(jo_snd_data,ptrs)
    v_snd = JaggedArray(v_snd_data,ptrs)
    gi_snd, gj_snd, jo_snd, v_snd
  end
  """
    Pushes to coo_entries_with_column_owner the tuples
    gi_rcv,gj_rcv,jo_rcv,v_rcv received from remote processes
  """
  function setup_rcv!(coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
    k_gi, k_gj, k_jo, k_v = coo_entries_with_column_owner
    current_n = length(k_gi)
    new_n = current_n + length(gi_rcv.data)
    println(typeof(k_gi))
    println(typeof(gi_rcv))
    resize!(k_gi,new_n)
    resize!(k_gj,new_n)
    resize!(k_jo,new_n)
    resize!(k_v,new_n)
    for p in 1:length(gi_rcv.data)
        k_gi[current_n+p] = gi_rcv.data[p]
        k_gj[current_n+p] = gj_rcv.data[p]
        k_jo[current_n+p] = jo_rcv.data[p]
        k_v[current_n+p] = v_rcv.data[p]
    end
  end
  part = linear_indices(row_partition)
  parts_snd, parts_rcv = assembly_neighbors(row_partition)
  coo_entries_with_column_owner = map(tuple,I,J,Jown,V)
  gi_snd, gj_snd, jo_snd, v_snd = map(setup_snd,part,parts_snd,row_partition,coo_entries_with_column_owner) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(gi_snd,graph)
  t2 = exchange(gj_snd,graph)
  t3 = exchange(jo_snd,graph)
  t4 = exchange(v_snd,graph)
  @async begin
      gi_rcv = fetch(t1)
      gj_rcv = fetch(t2)
      jo_rcv = fetch(t3)
      v_rcv = fetch(t4)
      map(setup_rcv!,coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
      I,J,Jown,V
  end
end

######################
@inline function Algebra._add_entries!(combine::Function,A,vs,is,js)
  println(typeof(A))
  println(typeof(vs))
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          vij = vs[li,lj]
          Algebra.add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

# test_sparse_matrix_assembler(assem,matdata,vecdata,data)
# A = allocate_matrix(assem,matdata)
A = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
symbolic_loop_matrix!(A,assem,matdata)
strategy = FESpaces.get_assembly_strategy(assem)
cellmat,_cellidsrows,_cellidscols = matdata[1][1],matdata[2][1],matdata[3][1]
cellidsrows = FESpaces.map_cell_rows(strategy,_cellidsrows)
cellidscols = FESpaces.map_cell_cols(strategy,_cellidscols)
@assert length(cellidscols) == length(cellidsrows)
rows_cache = array_cache(cellidsrows)
cols_cache = array_cache(cellidscols)
get_mat(a::Tuple) = a[1]
get_mat(a) = a
mat1 = get_mat(first(cellmat))
rows1 = getindex!(rows_cache,cellidsrows,1)
cols1 = getindex!(cols_cache,cellidscols,1)
touch! = TouchEntriesMap()
touch_cache = return_cache(touch!,A,mat1,rows1,cols1)
# FESpaces._symbolic_loop_matrix!(A,caches,cellidsrows,cellidscols,mat1)
rows = getindex!(rows_cache,cellidsrows,1)
cols = getindex!(cols_cache,cellidscols,1)
# evaluate!(touch_cache,touch!,A,mat1,rows,cols)
# add_entries!(+,A,nothing,rows,cols)
# Algebra._add_entries!(+,A,nothing,rows,cols)
for (lj,j) in enumerate(js)
  if j>0
    for (li,i) in enumerate(is)
      if i>0
        add_entry!(combine,A,nothing,i,j)
      end
    end
  end
end

dΩ = Measure(trian,degree)
form(u,v) = ∫(u*v)dΩ
matdata1 = collect_cell_matrix(V,V,form(u,v))
assem1 = SparseMatrixAssembler(T,Vector{Float64},V,V)
m11 = nz_counter(get_matrix_builder(assem1),(get_rows(assem1),get_cols(assem1)))
symbolic_loop_matrix!(m11,assem1,matdata1)
A1 = nz_allocation(m11)
