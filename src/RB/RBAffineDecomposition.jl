struct RBIntegrationDomain
  meas::Measure
  times::Vector{<:Real}
  idx::Vector{Int}

  function RBIntegrationDomain(meas::Measure,times::Vector{<:Real},idx::Vector{Int})
    new(meas,times,idx)
  end

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
    RBIntegrationDomain(red_meas,red_times,interp_idx_space)
  end
end

function Arrays.testvalue(::Type{RBIntegrationDomain},feop::PTFEOperator)
  test = get_test(feop)
  trian = get_triangulation(test)
  meas = Measure(trian,0)
  times = Vector{Real}(undef,0)
  idx = Vector{Int}(undef,0)
  RBIntegrationDomain(meas,times,idx)
end

struct RBAffineDecomposition{T}
  basis_space::Vector{Matrix{T}}
  basis_time::Matrix{T}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain

  function RBAffineDecomposition(
    basis_space::Vector{Matrix{T}},
    basis_time::Matrix{T},
    mdeim_interpolation::LU,
    integration_domain::RBIntegrationDomain) where T
    new{T}(basis_space,basis_time,mdeim_interpolation,integration_domain)
  end

  function RBAffineDecomposition(
    info::RBInfo,
    feop::PTFEOperator,
    snaps::PTArray,
    meas::Measure,
    times::Vector{<:Real},
    args...;
    kwargs...)

    basis_space,basis_time = compress(info,snaps;ϵ=info.ϵ)
    proj_bs,proj_bt = compress(basis_space,basis_time,args...;kwargs...)
    interp_idx_space = get_interpolation_idx(basis_space)
    interp_idx_time = get_interpolation_idx(basis_time)
    entire_interp_idx_space = recast_idx(basis_space,interp_idx_space)

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
end

function Arrays.testvalue(
  ::Type{RBAffineDecomposition{T}},
  feop::PTFEOperator) where T

  basis_space = Vector{Matrix{T}}(undef,0)
  basis_time = Matrix{T}(undef,0)
  mdeim_interpolation = lu(one(T))
  integration_domain = testvalue(IntegrationDomain,feop)
  RBAffineDecomposition(basis_space,basis_time,mdeim_interpolation,integration_domain)
end

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::RBSpace,
  snaps::Snapshots,
  μ::Table)

  nsnaps = info.nsnaps
  snapsθ = recenter(fesolver,snaps,μ)
  _snapsθ,_μ = snapsθ[1:nsnaps],μ[1:nsnaps]
  rhs = collect_compress_rhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  lhs = collect_compress_lhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  rhs,lhs
end

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::RBSpace,
  args...)

  times = get_times(fesolver)
  ress,meas = collect_residuals(fesolver,feop,args...)
  ad_res = compress_component(info,feop,ress,meas,times,rbspace)
  return ad_res
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::RBSpace,
  args...)

  times = get_times(fesolver)
  θ = fesolver.θ

  njacs = length(feop.jacs)
  ad_jacs = Vector{RBAlgebraicContribution}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : x-y
    jacs,meas = collect_jacobians(fesolver,feop,args...;i)
    ad_jacs[i] = compress_component(info,feop,jacs,meas,times,rbspace,rbspace;combine_projections)
  end
  return ad_jacs
end

function compress_component(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::Snapshots{T},
  meas::Vector{Measure},
  args...;
  kwargs...) where T

  contrib = RBAlgebraicContribution(T)
  map(eachindex(meas)) do i_meas
    si,mi = snaps[i_meas],meas[i_meas]
    ci = RBAffineDecomposition(info,feop,si,mi,args...;kwargs...)
    add_contribution!(contrib,mi,ci)
  end
end

function get_interpolation_idx(nzm::NnzMatrix)
  get_interpolation_idx(get_nonzero_val(nzm))
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

function compress(basis_space::NnzMatrix,::NnzMatrix,args...;kwargs...)
  compress_space(basis_space,args...),compress_time(args...;kwargs...)
end

function compress_space(
  basis_space::NnzMatrix,
  rbspace_row::RBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  entire_bs = recast(basis_space)
  compress(entire_bs_row,entire_bs)
end

function compress_space(
  basis_space::NnzMatrix,
  rbspace_row::RBSpace,
  rbspace_col::RBSpace)

  entire_bs_row = get_basis_space(rbspace_row)
  entire_bs_col = get_basis_space(rbspace_col)
  compress(entire_bs_row,entire_bs_col,basis_space)
end

function compress_time(rbspace_row::RBSpace,args...;kwargs...)
  get_basis_time(rbspace_row)
end

function compress_time(
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

  combine_projections(bt_proj,bt_proj_shift)
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

# Multifield interface

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  times::Vector{<:Real},
  rbspace::BlockRBSpace...;
  kwargs...)

  nfields = get_nfields(testitem(rbspace))
  times = get_times(fesolver)
  contrib = testvalue(RBBlockAlgebraicContribution{T},feop,(nfields,1))
  @inbounds for i_field = index_pairs(nfields,1)
    row,_ = i_field
    feop_i = filter_operator(feop,i_field)
    rbspace_i = map(x->filter_rbspace(x,row),rbspace)
    ress_i,meas_i = collect_residuals(feop_i,fesolver,args...)
    if iszero(ress_i)
      contrib.touched[row,1] = false
    else
      contrib.block[row,1] = compress_component(info,feop_i,ress_i,meas_i,times,rbspace_i)
    end
  end
  return contrib
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  times::Vector{<:Real},
  rbspace::BlockRBSpace...;
  kwargs...)

  njacs = length(feop.jacs)
  nfields = get_nfields(testitem(rbspace))
  times = get_times(fesolver)
  θ = fesolver.θ
  contrib = testvalue(RBBlockAlgebraicContribution{T},feop,(nfields,nfields))
  contribs = Vector{typeof(contrib)}(undef,njacs)
  @inbounds for i_field = index_pairs(nfields,nfields)
    row,col = i_field
    feop_i = filter_operator(feop,i_field)
    rbspace_i = map(x->filter_rbspace(x,i_field),rbspace)
    for i = 1:njacs
      combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : x-y
      jacs_i,meas_i = collect_jacobians(feop,fesolver,args...;i)
      if iszero(ress_i)
        contribs[i].touched[row,col] = false
      else
        contribs.block[row,col] = compress_component(
          info,
          feop_i,
          jacs_i,
          meas_i,
          times,
          rbspace_i...;
          combine_projections)
      end
    end
  end
  return contribs
end
