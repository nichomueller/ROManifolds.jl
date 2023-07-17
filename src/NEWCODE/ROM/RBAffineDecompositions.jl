struct RBIntegrationDomain
  meas::Measure
  idx::Vector{Int}

  function RBIntegrationDomain(
    component::SingleFieldSnapshots,
    trian::Triangulation,
    interp_idx::Vector{Int},
    cell_dof_ids::Table,
    order=1)

    degree = 2*order
    nonzero_idx = component.snaps.nonzero_idx
    nrows = component.snaps.nrows
    entire_interp_idx = nonzero_idx[interp_idx]
    entire_interp_rows,_ = from_vec_to_mat_idx(entire_interp_idx,nrows)
    red_integr_cells = find_cells(entire_interp_rows,cell_dof_ids)
    red_trian = view(trian,red_integr_cells)
    red_meas = Measure(red_trian,degree)
    new(red_meas,entire_interp_idx_space)
  end
end

struct TransientRBIntegrationDomain
  meas::Measure
  times::Vector{Float}
  idx::Vector{Int}

  function TransientRBIntegrationDomain(
    component::TransientSingleFieldSnapshots,
    trian::Triangulation,
    times::Vector{Float},
    interp_idx_space::Vector{Int},
    interp_idx_time::Vector{Int},
    cell_dof_ids::Table,
    order=1;
    st_mdeim=true)

    degree = 2*order
    nonzero_idx = component.snaps.nonzero_idx
    nrows = component.snaps.nrows
    entire_interp_idx_space = nonzero_idx[interp_idx_space]
    entire_interp_rows_space,_ = from_vec_to_mat_idx(entire_interp_idx_space,nrows)
    red_integr_cells = find_cells(entire_interp_rows_space,cell_dof_ids)
    red_trian = view(trian,red_integr_cells)
    red_meas = Measure(red_trian,degree)
    red_times = st_mdeim ? times[interp_idx_time] : times
    new(red_meas,red_times,entire_interp_idx_space)
  end
end

abstract type RBAffineDecompositions end

struct RBAffineDecomposition <: RBAffineDecompositions
  basis_space::Vector{<:AbstractMatrix}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain
end

struct TransientRBAffineDecomposition <: RBAffineDecompositions
  basis_space::Vector{<:AbstractMatrix}
  basis_time::Tuple{Vararg{AbstractArray}}
  mdeim_interpolation::LU
  integration_domain::TransientRBIntegrationDomain
end

struct ZeroRBAffineDecomposition <: RBAffineDecompositions
  proj::AbstractMatrix
end

for (Top,Tslv,Tsnp,Tsps,Tspm) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver),
  (:Snapshots,:TransientSnapshots),
  (:SingleFieldRBSpace,:TransientSingleFieldRBSpace),
  (:MultiFieldRBSpace,:TransientMultiFieldRBSpace))

  @eval begin
    function compress_residuals(
      feop::$Top,
      fesolver::$Tslv,
      rbspace::Union{$Tsps,$Tspm},
      s::$Tsnp,
      params::Table;
      kwargs...)

      trians = _collect_trian_res(feop)
      cres = RBResidualContribution()
      for trian in trians
        ad = compress_residuals(feop,fesolver,rbspace,s,params,trian;kwargs...)
        add_contribution!(cres,trian,ad)
      end
      cres
    end

    function compress_residuals(
      feop::$Top,
      fesolver::$Tslv,
      rbspace::Union{$Tsps,$Tspm},
      s::$Tsnp,
      params::Table,
      trian::Triangulation;
      kwargs...)

      nfields = length(rbspace)
      ad_res = Vector{RBAffineDecompositions}(undef,nfields)
      for row = 1:nfields
        filter = (row,1)
        ad_res[row] = compress_residuals(feop,fesolver,
          rbspace[row],s,params,trian,filter;kwargs...)
      end
      ad_res
    end

    function compress_residuals(
      feop::$Top,
      fesolver::$Tslv,
      rbspace::$Tsps,
      s::$Tsnp,
      params::Table,
      trian::Triangulation,
      filter::Tuple{Vararg{Int}};
      nsnaps=20,
      kwargs...)

      row, = filter
      sres = s[1:nsnaps]
      pres = params[1:nsnaps]
      cell_dof_ids = get_cell_dof_ids(feop.test[row],trian)
      order = get_order(feop.test[row])

      r = collect_residuals(feop,fesolver,sres,pres,trian,filter)
      compress_component(r,fesolver,trian,cell_dof_ids,order,rbspace;kwargs...)
    end

    function compress_jacobians(
      feop::$Top,
      fesolver::$Tslv,
      rbspace::Union{$Tsps,$Tspm},
      s::$Tsnp,
      params::Table;
      kwargs...)

      trians = _collect_trian_jac(feop)
      cjac = RBJacobianContribution()
      for trian in trians
        ad = compress_jacobians(feop,fesolver,rbspace,s,params,trian;kwargs...)
        add_contribution!(cjac,trian,ad)
      end
      cjac
    end

    function compress_jacobians(
      feop::$Top,
      fesolver::$Tslv,
      rbspace::Union{$Tsps,$Tspm},
      s::$Tsnp,
      params::Table,
      trian::Triangulation;
      kwargs...)

      nfields = length(rbspace)
      ad_jac = Matrix{RBAffineDecompositions}(undef,nfields,nfields)
      for row = 1:nfields, col = 1:nfields
        filter = (row,col)
        ad_jac[row,col] = compress_jacobians(feop,fesolver,
          (rbspace[row],rbspace[col]),s,params,trian,filter;kwargs...)
      end
      ad_jac
    end

    function compress_jacobians(
      feop::$Top,
      fesolver::$Tslv,
      rbspace::NTuple{2,$Tsps},
      s::$Tsnp,
      params::Table,
      trian::Triangulation,
      filter::Tuple{Vararg{Int}};
      nsnaps=20,
      kwargs...)

      row, = filter
      sjac = s[1:nsnaps]
      pjac = params[1:nsnaps]
      cell_dof_ids = get_cell_dof_ids(feop.test[row],trian)
      order = get_order(feop.test[row])

      j = collect_jacobians(feop,fesolver,sjac,pjac,trian,filter)
      compress_component(j,fesolver,trian,cell_dof_ids,order,rbspace...;kwargs...)
    end
  end

end

function compress_component(
  component::SingleFieldSnapshots,
  fesolver::FESolver,
  trian::Triangulation,
  cell_dof_ids::Table,
  order::Int,
  args...;
  kwargs...)

  bs = tpod(component;kwargs...)
  interp_idx = get_interpolation_idx(bs)
  integr_domain = RBIntegrationDomain(component,trian,interp_idx,cell_dof_ids,order)

  interp_bs = bs.nonzero_val[interp_idx,:]
  lu_interp_bs = lu(interp_bs)

  proj_bs = compress(fesolver,bs,args...)

  RBAffineDecomposition(proj_bs,lu_interp_bs,integr_domain)
end

function compress_component(
  component::TransientSingleFieldSnapshots,
  fesolver::ODESolver,
  trian::Triangulation,
  cell_dof_ids::Table,
  order::Int,
  args...;
  st_mdeim=true,
  kwargs...)

  times = get_times(fesolver)

  bs,bt = tpod(component,fesolver;ϵ)
  interp_idx_space,interp_idx_time = get_interpolation_idx(bs,bt)
  integr_domain = TransientRBIntegrationDomain(
    component,trian,times,interp_idx_space,interp_idx_time,cell_dof_ids,order;st_mdeim)

  interp_bs = bs.nonzero_val[interp_idx_space,:]
  interp_bt = bt.nonzero_val[interp_idx_time,:]
  interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
  lu_interp_bst = lu(interp_bst)

  proj_bs,proj_bt = compress(fesolver,bs,bt,args...)

  TransientRBAffineDecomposition(proj_bs,proj_bt,lu_interp_bst,integr_domain)
end

function compress_component(
  ::SingleFieldSnapshots{T,ZeroAffinity},
  ::FESolver,
  ::Triangulation,
  ::Table,
  ::Int,
  args...;
  kwargs...) where T

  proj = zero_compress(args...)
  ZeroRBAffineDecomposition(proj)
end

function compress_component(
  ::TransientSingleFieldSnapshots{T,ZeroAffinity},
  ::ODESolver,
  ::Triangulation,
  ::Table,
  ::Int,
  args...;
  kwargs...) where T

  proj = zero_compress(args...)
  ZeroRBAffineDecomposition(proj)
end

function get_interpolation_idx(basis_space::NnzArray,basis_time::NnzArray)
  idx_space = get_interpolation_idx(basis_space)
  idx_time = get_interpolation_idx(basis_time)
  idx_space,idx_time
end

function get_interpolation_idx(basis::NnzArray)
  get_interpolation_idx(basis.nonzero_val)
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
  ::FESolver,
  bs_component::NnzArray,
  args...)

  compress_space(bs_component,args...)
end

function compress(
  fesolver::ODESolver,
  bs_component::NnzArray,
  bt_component::NnzArray,
  args...)

  compress_space(bs_component,args...),compress_time(fesolver,bt_component,args...)
end

for Trb in (:SingleFieldRBSpace,:TransientSingleFieldRBSpace)

  @eval begin
    function compress_space(
      bs_component::NnzArray,
      rbspace_row::$Trb{<:AbstractMatrix})

      bs_row = get_basis_space(rbspace_row)
      entire_bs_row = recast(bs_row)
      entire_bs_component = recast(bs_component)
      map(eachcol(entire_bs_component)) do col
        cmat = reshape(col,:,1)
        entire_bs_row'*cmat
      end
    end

    function compress_space(
      bs_component::NnzArray,
      rbspace_row::$Trb{<:AbstractMatrix},
      rbspace_col::$Trb{<:AbstractMatrix})

      bs_row = get_basis_space(rbspace_row)
      bs_col = get_basis_space(rbspace_col)
      entire_bs_row = recast(bs_row)
      entire_bs_col = recast(bs_col)
      nbasis = size(bs_component,2)
      map(1:nbasis) do n
        entire_bs_row'*recast(bs_component,n)*entire_bs_col
      end
    end
  end
end

function compress_time(
  ::θMethod,
  bt_component::NnzArray,
  rbspace_row::TransientSingleFieldRBSpace{<:AbstractMatrix})

  bt = get_basis_time(rbspace_row)
  bt_component.nonzero_val,bt.nonzero_val
end

function compress_time(
  ::θMethod,
  bt_component::NnzArray,
  rbspace_row::TransientSingleFieldRBSpace{<:AbstractMatrix},
  rbspace_col::TransientSingleFieldRBSpace{<:AbstractMatrix})

  bt_row = get_basis_time(rbspace_row).nonzero_val
  bt_col = get_basis_time(rbspace_col).nonzero_val
  time_ndofs = size(bt_row,1)
  nt_row,nt_col = size(bt_row,2),size(bt_col,2)

  btbt = zeros(time_ndofs,nt_row,nt_col)
  btbt_shift = copy(btbt)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    btbt[:,it,jt] .= bt_row[:,it].*bt_col[:,jt]
    btbt_shift[2:time_ndofs,it,jt] .= (bt_row[2:time_ndofs,it].*
      bt_col[1:time_ndofs-1,jt])
  end

  bt_component.nonzero_val,btbt+btbt_shift
end

function zero_compress(
  rbspace_row::SingleFieldRBSpace{<:AbstractMatrix})

  bs_row = get_basis_space(rbspace_row)
  nbasis_row = size(bs_row,2)
  allocate_matrix(bs_row,nbasis_row,1)
end

function zero_compress(
  rbspace_row::TransientSingleFieldRBSpace{<:AbstractMatrix})

  bs_row = get_basis_space(rbspace_row)
  bt_row = get_basis_time(rbspace_row)
  nbasis_row = size(bs_row,2)*size(bt_row,2)
  allocate_matrix(bs_row,nbasis_row,1)
end

function zero_compress(
  rbspace_row::TransientSingleFieldRBSpace{<:AbstractMatrix},
  rbspace_col::TransientSingleFieldRBSpace{<:AbstractMatrix})

  bs_row = get_basis_space(rbspace_row)
  bt_row = get_basis_time(rbspace_row)
  bs_col = get_basis_space(rbspace_col)
  bt_col = get_basis_time(rbspace_col)
  nbasis_row = size(bs_row,2)*size(bt_row,2)
  nbasis_col = size(bs_col,2)*size(bt_col,2)
  allocate_matrix(bs_row,nbasis_row,nbasis_col)
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
