struct RBIntegrationDomain
  meas::Measure
  times::Vector{<:Real}
  idx::Vector{Int}

  function RBIntegrationDomain(
    feop::PTFEOperator,
    meas::Measure,
    times::Vector{<:Real},
    interp_idx_space::Vector{Int},
    interp_idx_time::Vector{Int},
    entire_interp_idx_space::Vector{Int};
    st_mdeim=false)

    trian = get_triangulation(meas)
    cell_dof_ids = get_cell_dof_ids(feop.test,trian)
    order = get_order(feop.test)
    degree = 2*order

    red_integr_cells = find_cells(entire_interp_idx_space,cell_dof_ids)
    red_trian = view(trian,red_integr_cells)
    red_meas = Measure(red_trian,degree)
    red_times = st_mdeim ? times[interp_idx_time] : times
    new(red_meas,red_times,interp_idx_space)
  end
end

struct RBAffineDecomposition{T}
  basis_space::Vector{Matrix{T}}
  basis_time::Vector{Matrix{T}}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain
end

function collect_compress_residual(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::ODESolver,
  rbspace::AbstractRBSpace,
  snaps::AbstractNnzMatrix,
  params::Table)

  cres = RBAlgebraicContribution()
  ad_res,meas = collect_compress_residual(info,feop,fesolver,rbspace,snaps,params)
  for (ad,m) in zip(ad_res,meas)
    add_contribution!(cres,m,ad)
  end
  return cres
end

function collect_compress_residual(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::ODESolver,
  rbspace::RBSpace,
  snaps::NnzMatrix,
  params::Table)

  nsnaps = info.nsnaps_system
  times = get_times(fesolver)
  ress,meas = collect_residuals(feop,fesolver,snaps,params;nsnaps)
  ad_res = if isa(meas,AbstractVector)
    map(eachindex(meas)) do i
      compress_component(info,ress[i],feop,meas[i],times,rbspace)
    end
  else
    compress_component(info,ress,feop,meas,times,rbspace)
  end
  return ad_res,meas
end

function collect_compress_residual(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::ODESolver,
  rbspace::BlockRBSpace,
  snaps::BlockNnzMatrix,
  params::Table)

  nsnaps = info.nsnaps_system
  times = get_times(fesolver)
  nfields = get_nfields(rbspace)
  all_idx = index_pairs(1:nfields,1)
  ress,meas = collect_residuals(filt_op,fesolver,snaps,params;nsnaps)
  ad_res = if isa(meas,AbstractVector)
    map(eachindex(meas)) do i_meas
      map(all_idx) do i_field
        feop_i = filter_operator(feop,i_field)
        rbspace_i = filter_rbspace(rbspace,i_field)
        compress_component(info,ress[i_meas],feop_i,meas[i_meas],times,rbspace_i)
      end
    end
  else
    map(all_idx) do i_field
      feop_i = filter_operator(feop,i_field)
      rbspace_i = filter_rbspace(rbspace,i_field)
      compress_component(info,ress,feop_i,meas,times,rbspace_i)
    end
  end
  return ad_res,meas
end

function collect_compress_jacobians(
  info::RBInfo,
  feop::PTFEOperator,
  args...)

  njacs = length(feop.jacs)
  cjacs = Vector{RBAlgebraicContribution}(undef,njacs)
  for i = 1:njacs
    cjac = collect_compress_jacobian(info,feop,args...;i)
    cjacs[i] = copy(cjac)
  end
  return cjacs
end

function collect_compress_jacobian(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::ODESolver,
  rbspace::AbstractRBSpace,
  snaps::AbstractNnzMatrix,
  params::Table;
  i=1)

  cjac = RBAlgebraicContribution()
  ad_jac,meas = collect_compress_jacobian(info,feop,fesolver,rbspace,snaps,params;i)
  for (ad,m) in zip(ad_jac,meas)
    add_contribution!(cjac,m,ad)
  end
  return cjac
end

function collect_compress_jacobian(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::ODESolver,
  rbspace::AbstractRBSpace,
  snaps::AbstractNnzMatrix,
  params::Table;
  i=1)

  nsnaps = info.nsnaps_system
  times = get_times(fesolver)
  function combine_projections(x,y)
    if i == 1
      fesolver.θ*x+(1-fesolver.θ)*y
    else
      x-y
    end
  end

  jacs,meas = collect_jacobians(feop,fesolver,snaps,params;nsnaps,i)
  compress_component(info,jacs,feop,meas,times,rbspace,rbspace;combine_projections)
end

# function collect_compress_jacobian(
#   info::RBInfo,
#   feop::PTFEOperator,
#   fesolver::ODESolver,
#   rbspace::BlockRBSpace,
#   snaps::MultiFieldSnapshots,
#   params::Table;
#   i=1)

#   nsnaps = info.nsnaps_system
#   nfields = get_nfields(rbspace)
#   all_idx = index_pairs(1:nfields,1:nfields)
#   times = get_times(fesolver)
#   function combine_projections(x,y)
#     if i == 1
#       fesolver.θ*x+(1-fesolver.θ)*y
#     else
#       x-y
#     end
#   end

#   ad_jac = lazy_map(all_idx) do filter
#     filt_op = filter_operator(feop,filter)
#     filt_rbspace = filter_rbspace(rbspace,filter[1]),filter_rbspace(rbspace,filter[2])

#     jacs = collect_jacobians(filt_op,fesolver,snaps,params,trian;nsnaps,i)
#     compress_component(info,jacs,filt_op,trian,times,filt_rbspace...;combine_projections)
#   end
#   return ad_jac
# end

function compress_component(
  info::RBInfo,
  snap::Vector{<:AbstractNnzMatrix},
  feop::PTFEOperator,
  meas::Vector{<:Measure},
  args...;
  kwargs...)

  map(eachindex(meas)) do i
    compress_component(info,snap[i],feop,meas[i],args...;kwargs...)
  end
end

function compress_component(
  info::RBInfo,
  snap::BlockNnzMatrix,
  feop::PTFEOperator,
  meas::Measure,
  times::Vector{<:Real},
  rbspace::BlockRBSpace...;
  kwargs...)

  nfields = get_nfields(snap)
  all_idx = index_pairs(1:nfields,1:nfields)
  map(all_idx) do idx
    row,col = idx
    feop_i = filter_operator(feop,filter)
    rbspace_i = map(filter_rbspace,rbspace,idx)
    compress_component(info,snap[col],feop_i,meas,times,rbspace_i...;kwargs...)
  end
end

function RBAffineDecomposition(
  info::RBInfo,
  snap::AbstractNnzMatrix,
  feop::PTFEOperator,
  meas::Measure,
  times::Vector{<:Real},
  args...;
  kwargs...)

  basis_space,basis_time = tpod(snap;ϵ=info.ϵ)
  proj_bs,proj_bt = compress(basis_space,basis_time,args...;kwargs...)
  interp_idx_space,interp_idx_time = get_interpolation_idx(basis_space,basis_time)
  entire_interp_idx_space = recast_index(basis_space,interp_idx_space)

  interp_bs = basis_space.nonzero_val[interp_idx_space,:]
  lu_interp = if info.st_mdeim
    interp_bt = basis_time.nonzero_val[interp_idx_time,:]
    interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
    lu(interp_bst)
  else
    lu(interp_bs)
  end

  integr_domain = RBIntegrationDomain(
    feop,
    meas,
    times,
    interp_idx_space,
    interp_idx_time,
    entire_interp_idx_space;
    info.st_mdeim)

  RBAffineDecomposition(proj_bs,proj_bt,lu_interp,integr_domain)
end

function get_interpolation_idx(basis_space::NnzMatrix,basis_time::NnzMatrix)
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
  basis_space::NnzMatrix,
  basis_time::NnzMatrix,
  args...;
  kwargs...)

  compress_space(basis_space,args...),compress_time(basis_time,args...;kwargs...)
end

function compress_space(
  basis_space::NnzMatrix,
  rbspace_row::RBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  entire_bs = recast(basis_space)
  map(eachcol(entire_bs)) do col
    cmat = reshape(col,:,1)
    entire_bs_row'*cmat
  end
end

function compress_space(
  basis_space::NnzMatrix,
  rbspace_row::RBSpace,
  rbspace_col::RBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  entire_bs_col = get_basis_space(rbspace_col)
  map(axes(basis_space,2)) do n
    entire_bs_row'*recast(basis_space,n)*entire_bs_col
  end
end

function compress_time(
  basis_time::NnzMatrix,
  rbspace_row::RBSpace,
  args...)

  bt = get_basis_time(rbspace_row)
  [basis_time.nonzero_val,bt]
end

function compress_time(
  basis_time::NnzMatrix,
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T};
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
  [basis_time.nonzero_val,combine_bt_proj]
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
