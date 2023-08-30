struct RBIntegrationDomain
  meas::Measure
  times::Vector{<:Real}
  idx::Vector{Int}

  function RBIntegrationDomain(
    feop::ParamTransientFEOperator,
    snap::Snapshots,
    trian::Triangulation,
    times::Vector{<:Real},
    interp_idx_space::Vector{Int},
    interp_idx_time::Vector{Int};
    st_mdeim=true)

    cell_dof_ids = get_cell_dof_ids(feop.test,trian)
    order = get_order(feop.test)
    degree = 2*order

    nonzero_idx = snap.snaps.nonzero_idx
    nrows = snap.snaps.nrows
    entire_interp_idx_space = nonzero_idx[interp_idx_space]
    entire_interp_rows_space,_ = from_vec_to_mat_idx(entire_interp_idx_space,nrows)
    red_integr_cells = find_cells(entire_interp_rows_space,cell_dof_ids)
    red_trian = view(trian,red_integr_cells)
    red_meas = Measure(red_trian,degree)
    red_times = st_mdeim ? times[interp_idx_time] : times
    new(red_meas,red_times,interp_idx_space)
  end
end

struct RBAffineDecomposition
  basis_space::Vector{<:AbstractMatrix}
  basis_time::Tuple{Vararg{AbstractArray}}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain
end

function compress_residuals(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rbspace::RBSpace,
  snaps::Snapshots,
  params::Table;
  kwargs...)

  trians = collect_trian_res(feop)
  cres = RBAlgebraicContribution()
  for trian in trians
    ad = compress_residuals(feop,fesolver,rbspace,snaps,params,trian;kwargs...)
    add_contribution!(cres,trian,ad)
  end
  return cres
end

function compress_residuals(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rbspace::SingleFieldRBSpace,
  snaps::SingleFieldSnapshots,
  params::Table,
  trian::Triangulation;
  nsnaps=20,kwargs...)

  times = get_times(fesolver)
  ress = collect_residuals(feop,fesolver,snaps,params,trian;nsnaps)
  ad_res = compress_residuals(ress,feop,trian,times,rbspace;kwargs...)
  return ad_res
end

function compress_residuals(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rbspace::MultiFieldRBSpace,
  snaps::MultiFieldSnapshots,
  params::Table,
  trian::Triangulation;
  nsnaps=20,kwargs...)

  nfields = get_nfields(rbspace)
  all_idx = index_pairs(1:nfields,1)
  ad_res = lazy_map(all_idx) do filter
    filt_op = filter_operator(feop,filter)
    filt_rbspace = filter_rbspace(rbspace,filter[1])
    times = get_times(fesolver)

    ress = collect_residuals(filt_op,fesolver,snaps,params,trian;nsnaps)
    compress_residuals(ress,filt_op,trian,times,filt_rbspace;kwargs...)
  end
  return ad_res
end

function compress_jacobians(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rbspace::RBSpace,
  snaps::Snapshots,
  params::Table;
  kwargs...)

  trians = collect_trian_jac(feop)
  cres = RBAlgebraicContribution()
  for trian in trians
    ad = compress_jacobians(feop,fesolver,rbspace,snaps,params,trian;kwargs...)
    add_contribution!(cres,trian,ad)
  end
  return cres
end

function compress_jacobians(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rbspace::SingleFieldRBSpace,
  snaps::SingleFieldSnapshots,
  params::Table,
  trian::Triangulation;
  i=1,nsnaps=20,kwargs...)

  times = get_times(fesolver)
  function combine_projections(x,y)
    if i == 1
      fesolver.θ*x+(1-fesolver.θ)*y
    else
      x-y
    end
  end

  jacs = collect_jacobians(feop,fesolver,snaps,params,trian;nsnaps,i)
  ad_jac = compress_jacobians(jacs,feop,trian,times,rbspace,rbspace;combine_projections,kwargs...)
  return ad_jac
end

function compress_jacobians(
  feop::ParamTransientFEOperator,
  fesolver::ODESolver,
  rbspace::MultiFieldRBSpace,
  snaps::MultiFieldSnapshots,
  params::Table,
  trian::Triangulation;
  i=1,nsnaps=20,kwargs...)

  nfields = get_nfields(rbspace)
  all_idx = index_pairs(1:nfields,1:nfields)
  times = get_times(fesolver)
  function combine_projections(x,y)
    if i == 1
      fesolver.θ*x+(1-fesolver.θ)*y
    else
      x-y
    end
  end

  ad_jac = lazy_map(all_idx) do filter
    filt_op = filter_operator(feop,filter)
    filt_rbspace = filter_rbspace(rbspace,filter[1]),filter_rbspace(rbspace,filter[2])

    jacs = collect_jacobians(filt_op,fesolver,snaps,params,trian;nsnaps,i)
    compress_jacobians(jacs,filt_op,trian,times,filt_rbspace...;combine_projections,kwargs...)
  end
  return ad_jac
end

function compress_component(
  snap::Snapshots,
  feop::ParamTransientFEOperator,
  trian::Triangulation,
  times::Vector{Real},
  args...;
  st_mdeim=true,ϵ=1e-4,kwargs...)

  basis_space,basis_time = tpod(snap;ϵ)
  interp_idx_space,interp_idx_time = get_interpolation_idx(basis_space,basis_time)

  interp_bs = basis_space.nonzero_val[interp_idx_space,:]
  lu_interp = if st_mdeim
    interp_bt = basis_time.nonzero_val[interp_idx_time,:]
    interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
    lu(interp_bst)
  else
    lu(interp_bs)
  end

  integr_domain = RBIntegrationDomain(feop,snap,trian,times,interp_idx_space,
    interp_idx_time;st_mdeim)
  proj_bs,proj_bt = compress(basis_space,basis_time,args...;kwargs...)
  GenericRBAffineDecomposition(proj_bs,proj_bt,lu_interp,integr_domain)
end

function get_interpolation_idx(basis_space::NnzArray,basis_time::NnzArray)
  idx_space = get_interpolation_idx(basis_space.nonzero_val)
  idx_time = get_interpolation_idx(basis_time.nonzero_val)
  idx_space,idx_time
end

function get_interpolation_idx(basis::AbstractMatrix)
  n = size(basis,2)
  idx = zeros(Int,n)
  idx[1] = argmax(abs.(basis[:,1]))
  if n > 1
    @inbounds for i = 2:n
      proj = basis[:,1:i-1]*(basis[idx[1:i-1],1:i-1] \ basis[idx[1:i-1],i])
      res = basis[:,i] - proj
      idx[i] = argmax(abs.(res))
    end
  end
  unique(idx)
end

function compress(
  basis_space::NnzArray,
  basis_time::NnzArray,
  args...;
  kwargs...)

  compress_space(basis_space,args...),compress_time(basis_time,args...;kwargs...)
end

function compress_space(
  basis_space::NnzArray,
  rbspace_row::SingleFieldRBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  entire_bs = recast(basis_space)
  lazy_map(eachcol(entire_bs)) do col
    cmat = reshape(col,:,1)
    entire_bs_row'*cmat
  end
end

function compress_space(
  basis_space::NnzArray,
  rbspace_row::SingleFieldRBSpace,
  rbspace_col::SingleFieldRBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  entire_bs_col = get_basis_space(rbspace_col)
  lazy_map(axes(basis_space,2)) do n
    entire_bs_row'*recast(basis_space,n)*entire_bs_col
  end
end

function compress_time(
  basis_time::NnzArray,
  rbspace_row::SingleFieldRBSpace,
  args...)

  bt = get_basis_time(rbspace_row)
  basis_time.nonzero_val,bt
end

function compress_time(
  basis_time::NnzArray,
  rbspace_row::SingleFieldRBSpace{T},
  rbspace_col::SingleFieldRBSpace{T};
  combine_projections=(x,y)->x) where T

  bt_row = get_basis_time(rbspace_row)
  bt_col = get_basis_time(rbspace_col)
  time_ndofs = size(bt_row,1)
  nt_row = size(bt_row,2)
  nt_col = size(bt_col,2)

  bt_proj = zeros(T,time_ndofs,nt_row,nt_col)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    bt_proj[:,it,jt] .= bt_row[:,it].*bt_col[:,jt]
    bt_proj_shift[2:end,it,jt] .= bt_row[2:end,it].*bt_col[1:end-1,jt]
  end

  combine_bt_proj = combine_projections(bt_proj,bt_proj_shift)
  basis_time.nonzero_val,combine_bt_proj
end

function find_cells(idx::Vector{Int},cell_dof_ids)
  find_cells(Val{length(idx)>length(cell_dof_ids)}(),idx,cell_dof_ids)
end

function find_cells(::Val{true},idx::Vector{Int},cell_dof_ids)
  cells = Int[]
  for cell = eachindex(cell_dof_ids)
    if !isempty(intersect(idx,abs.(cell_dof_ids[cell])))
      append!(cells,cell)
    end
  end
  unique(cells)
end

function find_cells(::Val{false},idx::Vector{Int},cell_dof_ids)
  cells = Vector{Int}[]
  for i = idx
    cell = findall(x->!isempty(intersect(abs.(x),i)),cell_dof_ids)
    cells = isempty(cell) ? cells : push!(cells,cell)
  end
  unique(reduce(vcat,cells))
end
