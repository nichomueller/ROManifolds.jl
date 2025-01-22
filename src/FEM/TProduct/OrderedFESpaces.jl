struct OrderedFESpace{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
  cell_dofs_ids::AbstractArray
end

function OrderedFESpace(f::SingleFieldFESpace)
  cell_dofs_ids = get_ordered_cell_dof_ids(f)
  return OrderedFESpace(f,cell_dofs_ids)
end

# FESpace interface

FESpaces.ConstraintStyle(::Type{<:OrderedFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::OrderedFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::OrderedFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::OrderedFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::OrderedFESpace) = f.cell_dofs_ids

FESpaces.get_fe_basis(f::OrderedFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::OrderedFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::OrderedFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::OrderedFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::OrderedFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::OrderedFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::OrderedFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::OrderedFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::OrderedFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::OrderedFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_vector_type(f::OrderedFESpace) = get_vector_type(f.space)

function FESpaces.scatter_free_and_dirichlet_values(f::OrderedFESpace,fv,dv)
  scatter_free_and_dirichlet_values(f.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(f::OrderedFESpace,cv)
  gather_free_and_dirichlet_values(remove_layer(f),cv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::OrderedFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,remove_layer(f),cv)
end

struct OIdsToIds{T,A<:AbstractVector{<:Integer}} <: AbstractVector{T}
  indices::Vector{T}
  terms::A
end

Base.size(a::OIdsToIds) = size(a.indices)
Base.getindex(a::OIdsToIds,i::Integer) = getindex(a.indices,i)
Base.setindex!(a::OIdsToIds,v,i::Integer) = setindex!(a.indices,v,i)

struct DofsToODofs{D,P,V} <: Map
  b::LagrangianDofBasis{P,V}
  pdof_to_dof::Vector{Int32}
  node_and_comps_to_odof::Array{V,D}
  orders::NTuple{D,Int}
end

function DofsToODofs(
  b::LagrangianDofBasis,
  node_and_comps_to_odof::AbstractArray,
  orders::Tuple)

  pdof_to_dof = _local_pdof_to_dof(b,orders)
  DofsToODofs(b,pdof_to_dof,node_and_comps_to_odof,orders)
end

function DofsToODofs(fe_dof_basis::Fill{<:LagrangianDofBasis},args...)
  DofsToODofs(testitem(fe_dof_basis),args...)
end

function DofsToODofs(fe_dof_basis::AbstractVector{<:Dof},args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function get_ndofs(k::DofsToODofs{D,P}) where {D,P}
  ncomps = num_components(P)
  nnodes = length(k.node_and_comps_to_odof)
  ncomps*nnodes
end

function Arrays.return_cache(k::DofsToODofs{D},cell::CartesianIndex{D}) where D
  local_ndofs = length(k.pdof_to_dof)
  odofs = OIdsToIds(zeros(Int32,local_ndofs),k.pdof_to_dof)
  return odofs
end

function Arrays.evaluate!(cache,k::DofsToODofs{D},cell::CartesianIndex{D}) where D
  first_new_node = k.orders .* (Tuple(cell) .- 1) .+ 1
  onodes_range = map(enumerate(first_new_node)) do (i,ni)
    ni:ni+k.orders[i]
  end
  local_comps_to_odofs = view(k.node_and_comps_to_odof,onodes_range...)
  local_nnodes = length(k.b.node_and_comp_to_dof)
  for (node,comps_to_odof) in enumerate(local_comps_to_odofs)
    for comp in k.b.dof_to_comp
      odof = comps_to_odof[comp]
      cache[node+(comp-1)*local_nnodes] = odof
    end
  end
  return cache
end

struct LocalNnzIdsToGlobaNnzIds{D,Pgr,Pgc,Plr,Plc,Vgr,Vgc} <: Map
  global_rows::DofsToODofs{D,Pgr,Vgr}
  global_cols::DofsToODofs{D,Pgc,Vgc}
  local_rows::NTuple{D,DofsToODofs{1,Plr,Int}}
  local_cols::NTuple{D,DofsToODofs{1,Plc,Int}}
end

get_global_nrows(k::LocalNnzIdsToGlobaNnzIds) = get_ndofs(k.global_rows)
get_global_ncols(k::LocalNnzIdsToGlobaNnzIds) = get_ndofs(k.global_cols)

get_local_nrows(k::LocalNnzIdsToGlobaNnzIds) = map(get_ndofs,k.local_rows)
get_local_ncols(k::LocalNnzIdsToGlobaNnzIds) = map(get_ndofs,k.local_cols)

function Arrays.return_cache(k::LocalNnzIdsToGlobaNnzIds{D,Pgr,Pgc},cell::CartesianIndex{D}) where {D,Pgr,Pgc}
  gcache_rows = return_cache(k.global_rows,cell)
  gcache_cols = return_cache(k.global_cols,cell)
  lcaches_rows = map(return_cache,k.local_rows,CartesianIndex.(cell.I))
  lcaches_cols = map(return_cache,k.local_cols,CartesianIndex.(cell.I))

  ncomps = num_components(Pgr)*num_components(Pgr)
  rows_no_comps = get_local_nrows(k)
  cols_no_comps = get_local_ncols(k)
  lnrows_no_comps = prod(length.(lcaches_rows))
  lncols_no_comps = prod(length.(lcaches_cols))

  gnnz_index = zeros(Int,lnrows_no_comps,lncols_no_comps,ncomps)
  lnnz_indices = zeros(VectorValue{D,Int},lnrows_no_comps,lncols_no_comps,ncomps)
  return (gnnz_index,lnnz_indices,gcache_rows,gcache_cols,lcaches_rows,lcaches_cols,
    rows_no_comps,cols_no_comps)
end

function Arrays.evaluate!(cache,k::LocalNnzIdsToGlobaNnzIds{D,Pgr,Pgc},cell::CartesianIndex{D}) where {D,Pgr,Pgc}
  (gnnz_index,lnnz_indices,gcache_rows,gcache_cols,lcaches_rows,lcaches_cols,
    rows_no_comps,cols_no_comps) = cache
  godofs_row = evaluate!(gcache_rows,k.global_rows,cell)
  godofs_col = evaluate!(gcache_cols,k.global_cols,cell)
  lodofs_rows = map(evaluate!,lcaches_rows,k.local_rows,CartesianIndex.(cell.I))
  lodofs_cols = map(evaluate!,lcaches_cols,k.local_cols,CartesianIndex.(cell.I))
  nrows_no_comps = prod(rows_no_comps)
  ncols_no_comps = prod(cols_no_comps)
  ncomps_row = num_components(Pgr)
  ncomps_col = num_components(Pgr)
  lnrows_no_comps = size(gnnz_index,1)
  lncols_no_comps = size(gnnz_index,2)
  for (row,godof_row) in enumerate(godofs_row)
    row_no_comp = fast_index(row,lnrows_no_comps)
    gnode_row,comp_row = _fast_and_slow_index(godof_row,nrows_no_comps)
    lnodes_row = _index_to_d_indices(gnode_row,rows_no_comps)
    for (col,godof_col) in enumerate(godofs_col)
      col_no_comp = fast_index(col,lncols_no_comps)
      gnode_col,comp_col = _fast_and_slow_index(godof_col,ncols_no_comps)
      lnodes_col = _index_to_d_indices(gnode_col,cols_no_comps)
      g_nnz_i = gnode_row + (gnode_col-1)*nrows_no_comps
      l_nnz_i = lnodes_row .+ (lnodes_col.-1).*rows_no_comps
      comp = comp_row+(comp_col-1)*ncomps_row
      gnnz_index[row_no_comp,col_no_comp,comp] = g_nnz_i
      lnnz_indices[row_no_comp,col_no_comp,comp] = VectorValue(l_nnz_i)
    end
  end
  return gnnz_index,lnnz_indices
end

# Assembly-related functions

@inline function Algebra.add_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)
  add_ordered_entries!(combine,A,vs,is,js)
end

for T in (:Any,:(Algebra.ArrayCounter))
  @eval begin
    @inline function Algebra.add_entries!(combine::Function,A::$T,vs,is::OIdsToIds)
      add_ordered_entries!(combine,A,vs,is)
    end
  end
end

@inline function add_ordered_entries!(combine::Function,A,vs::Nothing,is::OIdsToIds,js::OIdsToIds)
  Algebra._add_entries!(combine,A,vs,is.indices,js.indices)
end

@inline function add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)
  for (lj,j) in enumerate(js)
    if j>0
      ljp = js.terms[lj]
      for (li,i) in enumerate(is)
        if i>0
          lip = is.terms[li]
          vij = vs[lip,ljp]
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function add_ordered_entries!(combine::Function,A,vs::Nothing,is::OIdsToIds)
  Algebra._add_entries!(combine,A,vs,is.indices)
end

@inline function add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds)
  for (li,i) in enumerate(is)
    if i>0
      lip = is.terms[li]
      vi = vs[lip]
      add_entry!(A,vi,i)
    end
  end
  A
end

# utils

function get_ordered_cell_dof_ids(space::SingleFieldFESpace)
  cell_dofs_ids = get_cell_dof_ids(space)
  fe_dof_basis = get_data(get_fe_dof_basis(space))
  orders = get_polynomial_orders(space)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  tface_to_mface = get_tface_to_mface(trian)
  get_ordered_cell_dof_ids(model,cell_dofs_ids,fe_dof_basis,tface_to_mface,orders)
end

function get_ordered_cell_dof_ids(
  model::CartesianDiscreteModel,
  cell_dofs_ids::AbstractArray,
  fe_dof_basis::AbstractArray,
  tface_to_mface::AbstractVector,
  orders::Tuple)

  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition
  cells = CartesianIndices(ncells)
  tcells = view(cells,tface_to_mface)
  onodes = LinearIndices(orders .* ncells .+ 1 .- periodic)

  node_and_comps_to_odof = get_ordered_dof_ids(cell_dofs_ids,fe_dof_basis,cells,onodes,orders)
  k = DofsToODofs(fe_dof_basis,node_and_comps_to_odof,orders)
  lazy_map(k,tcells)
end

function get_ordered_cell_dof_ids(model::DiscreteModel,args...)
  @notimplemented "Background model must be cartesian for dof reordering"
end

cubic_polytope(::Val{d}) where d = @abstractmethod
cubic_polytope(::Val{1}) = SEGMENT
cubic_polytope(::Val{2}) = QUAD
cubic_polytope(::Val{3}) = HEX

function get_ordered_dof_ids(args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function get_ordered_dof_ids(
  cell_dofs_ids::AbstractArray,
  fe_dof_basis::Fill{<:LagrangianDofBasis},
  args...)

  get_ordered_dof_ids(cell_dofs_ids,testitem(fe_dof_basis),args...)
end

function get_ordered_dof_ids(
  cell_dofs_ids::AbstractArray,
  fe_dof_basis::LagrangianDofBasis{P,V},
  cells::CartesianIndices{D},
  onodes::LinearIndices{D},
  orders::NTuple{D,Int}
  ) where {D,P,V}

  cache = array_cache(cell_dofs_ids)
  p = cubic_polytope(Val(D))
  terms = DofMaps._get_terms(p,orders)
  nnodes = length(onodes)
  ncomps = num_components(V)
  ndofs = nnodes*ncomps
  vec_odofs = zeros(eltype(V),ndofs)
  for (icell,cell) in enumerate(cells)
    first_new_node = orders .* (Tuple(cell) .- 1) .+ 1
    onodes_range = map(enumerate(first_new_node)) do (i,ni)
      ni:ni+orders[i]
    end
    onodes_cell = view(onodes,onodes_range...)
    cell_dofs = getindex!(cache,cell_dofs_ids,icell)
    for node in 1:length(onodes_cell)
      comp_to_dof = fe_dof_basis.node_and_comp_to_dof[node]
      t = terms[node]
      onode = onodes_cell[t]
      for comp in 1:ncomps
        dof = cell_dofs[comp_to_dof[comp]]
        odof = onode + (comp-1)*nnodes
        vec_odofs[odof] = sign(dof)
      end
    end
  end

  nfree = 0
  ndiri = 0
  for (i,odof) in enumerate(vec_odofs)
    if odof > 0
      nfree += 1
      vec_odofs[i] = nfree
    else
      ndiri -= 1
      vec_odofs[i] = ndiri
    end
  end

  _get_node_and_comps_to_odof(fe_dof_basis,vec_odofs,onodes)
end

function get_ordered_dof_ids(
  cell_dofs_ids::AbstractArray,
  fe_basis::AbstractVector{<:Dof},
  args...)

  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function _get_node_and_comps_to_odof(
  ::LagrangianDofBasis{P,V},
  vec_odofs,
  onodes
  ) where {P,V}

  reshape(vec_odofs,size(onodes))
end

function _get_node_and_comps_to_odof(
  ::LagrangianDofBasis{P,V},
  vec_odofs,
  onodes
  ) where {P,V<:MultiValue}

  nnodes = length(onodes)
  ncomps = num_components(V)
  odofs = zeros(V,size(onodes))
  m = zero(Mutable(V))
  for onode in 1:nnodes
    for comp in 1:ncomps
      odof = onode + (comp-1)*nnodes
      m[comp] = vec_odofs[odof]
    end
    odofs[onode] = m
  end
  return odofs
end

function _local_pdof_to_dof(fe_dof_basis::AbstractVector{<:Dof},args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function _local_pdof_to_dof(fe_dof_basis::Fill{<:LagrangianDofBasis},orders::NTuple{D,Int}) where D
  _local_pdof_to_dof(testitem(fe_dof_basis),orders)
end

function _local_pdof_to_dof(b::LagrangianDofBasis,orders::NTuple{D,Int}) where D
  nnodes = length(b.node_and_comp_to_dof)
  ndofs = length(b.dof_to_comp)

  p = cubic_polytope(Val(D))
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  node_to_pnode = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  node_to_pnode_linear = LinearIndices(orders.+1)[node_to_pnode]

  pdof_to_dof = zeros(Int32,ndofs)
  for (inode,ipnode) in enumerate(node_to_pnode_linear)
    for icomp in b.dof_to_comp
      local_shift = (icomp-1)*nnodes
      pdof_to_dof[local_shift+ipnode] = local_shift + inode
    end
  end

  return pdof_to_dof
end

function get_d_sparse_dofs_to_sparse_dofs(trial::OrderedFESpace,test::OrderedFESpace)
  cellrows = get_cell_dof_ids(test,trian)
  cellcols = get_cell_dof_ids(trial,trian)
end

function get_d_sparse_dofs_to_sparse_dofs(
  trial::OrderedFESpace,
  test::OrderedFESpace,
  trian::Triangulation)

  desc = get_cartesian_descriptor(model)
  ncells = desc.partition
  cells = CartesianIndices(ncells)

  tface_to_mface = get_tface_to_mface(trian)
  tcells = view(cells,tface_to_mface)

  cellrows = get_cell_dof_ids(test,trian)
  cellcols = get_cell_dof_ids(trial,trian)
  rowmap = get_odof_map(cellrows)
  colmap = get_odof_map(cellcols)
  global_nrows = num_free_dofs(test)
  k = RowColToNnzIndex(rowmap,colmap,global_nrows)
  cache = array_cache(k)

  for icell in tface_to_mface
    cell = cells[icell]
    nnz_index = getindex!(cache,k,cell,icell)
  end

end

function _d_rows_columns_to_index(
  local_rows::Vector{CartesianIndex{D}},
  local_cols::Vector{CartesianIndex{D}},
  global_row::Integer,
  global_col::Integer,
  ) where D


  id = CartesianIndex((rows_1d[d],cols_1d[d]))
  findfirst(id==all_id_d)
end

function _fast_and_slow_index(i::Integer,s::Integer)
  fast_index(i,s),slow_index(i,s)
end

function _index_to_d_indices(i::Integer,s2::NTuple{2,Integer})
  _fast_and_slow_index(i,s2[1])
end

function _index_to_d_indices(i::Integer,sD::NTuple{D,Integer}) where D
  sD_minus_1 = sD[1:end-1]
  nD_minus_1 = prod(sD_minus_1)
  iD = slow_index(i,nD_minus_1)
  (_index_to_d_indices(i,sD_minus_1)...,iD)
end
