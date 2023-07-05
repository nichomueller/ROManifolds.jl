struct RBIntegrationDomain
  meas::Measure
  idx::Vector{Int}

  function RBIntegrationDomain(
    component::SingleFieldSnapshots,
    trian::Triangulation,
    interp_idx::Vector{Int},
    cell_dof_ids,
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
    component::SingleFieldSnapshots,
    trian::Triangulation,
    times::Vector{Float},
    interp_idx_space::Vector{Int},
    interp_idx_time::Vector{Int},
    cell_dof_ids,
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
  basis_space::Vector{<:AbstractArray}
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain
end

struct TransientRBAffineDecomposition <: RBAffineDecompositions
  basis_space::Vector{<:AbstractArray}
  basis_time::Tuple{Vararg{AbstractMatrix}}
  mdeim_interpolation::LU
  integration_domain::TransientRBIntegrationDomain
end

struct ZeroRBAffineDecomposition <: RBAffineDecompositions end

for (Top,Tslv,Tsps,Tspm) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver),
  (:SingleFieldRBSpace,:TransientSingleFieldRBSpace),
  (:MultiFieldRBSpace,:TransientMultiFieldRBSpace))

  @eval begin
    function compress_residuals(
      feop::$Top,
      fesolver::$Tslv,
      args...;
      kwargs...)

      trians = _collect_trian_res(feop)
      cres = RBResidualContribution()
      for trian in trians
        ad_res = compress_residuals(feop,fesolver,trian,args...;kwargs...)
        add_contribution!(cres,trian,ad_res)
      end
      cres
    end

    function compress_residuals(
      feop::$Top,
      fesolver::$Tslv,
      trian::Triangulation,
      rbspace::Union{$Tsps,$Tspm},
      s::Snapshots,
      params::Table;
      kwargs...)

      nfields = get_nfields(rbspace)
      ad_res = Vector{RBAffineDecompositions}(undef,nfields)
      for row = 1:nfields
        ad_res[row] = compress_residuals(feop,fesolver,trian,
          rbspace[row],s,params,(row,1);kwargs...)
      end
      ad_res
    end

    function compress_residuals(
      feop::$Top,
      fesolver::$Tslv,
      trian::Triangulation,
      rbspace::$Tsps,
      s::Snapshots,
      params::Table,
      filter::Tuple{Vararg{Int}};
      nsnaps=20,
      kwargs...)

      sres = get_datum(s[1:nsnaps])
      pres = params[1:nsnaps]
      cell_dof_ids = get_cell_dof_ids(feop.test[filter[1]],trian)
      order = get_order(feop.test[filter[1]])

      vecdata = _vecdata_residual(feop,fesolver,sres,pres,filter,trian)
      r = generate_residuals(feop,fesolver,pres,vecdata)
      compress_component(r,fesolver,trian,cell_dof_ids,order,rbspace;kwargs...)
    end

    function compress_jacobians(
      feop::$Top,
      fesolver::$Tslv,
      args...;
      kwargs...)

      trians = _collect_trian_jac(feop)
      cjac = RBJacobianContribution()
      for trian in trians
        ad_jac = compress_jacobians(feop,fesolver,trian,args...;kwargs...)
        add_contribution!(cjac,trian,ad_jac)
      end
      cjac
    end

    function compress_jacobians(
      feop::$Top,
      fesolver::$Tslv,
      trian::Triangulation,
      rbspace::Union{$Tsps,$Tspm},
      s::Snapshots,
      params::Table;
      kwargs...)

      nfields = get_nfields(rbspace)
      ad_jac = Matrix{RBAffineDecompositions}(undef,nfields,nfields)
      for row = 1:nfields, col = 1:nfields
        ad_jac[row,col] = compress_jacobians(feop,fesolver,trian,
          (rbspace[row],rbspace[col]),s,params,(row,col);kwargs...)
      end
      ad_jac
    end

    function compress_jacobians(
      feop::$Top,
      fesolver::$Tslv,
      trian::Triangulation,
      rbspace::NTuple{2,$Tsps},
      s::Snapshots,
      params::Table,
      filter=(1,1);
      nsnaps=20,
      kwargs...)

      sjac = get_datum(s[1:nsnaps])
      pjac = params[1:nsnaps]
      cell_dof_ids = get_cell_dof_ids(feop.test[filter[1]],trian)
      order = get_order(feop.test[filter[1]])

      matdata = _matdata_jacobian(feop,fesolver,sjac,pjac,filter,trian)
      j = generate_jacobians(feop,fesolver,pjac,matdata)
      compress_component(j,fesolver,trian,cell_dof_ids,order,rbspace...;kwargs...)
    end
  end

end

function compress_component(
  component::SingleFieldSnapshots{T,A} where T,
  fesolver::FESolver,
  trian::Triangulation,
  cell_dof_ids::Table,
  order::Int,
  args...;
  kwargs...) where {A<:Union{ParamAffinity,NonAffinity}}

  bs = tpod(component;kwargs...)
  interp_idx = get_interpolation_idx(bs)
  integr_domain = RBIntegrationDomain(component,trian,interp_idx,cell_dof_ids,order)

  interp_bs = bs.nonzero_val[interp_idx,:]
  lu_interp_bs = lu(interp_bs)

  proj_bs = compress(fesolver,bs,args...)

  RBAffineDecomposition(proj_bs,lu_interp_bs,integr_domain)
end

function compress_component(
  component::SingleFieldSnapshots{T,A} where T,
  fesolver::ODESolver,
  trian::Triangulation,
  cell_dof_ids::Table,
  order::Int,
  args...;
  st_mdeim=true,
  kwargs...) where {A<:Union{ParamAffinity,TimeAffinity,ParamTimeAffinity,NonAffinity}}

  times = get_times(fesolver)

  bs,bt = transient_tpod(component,fesolver;ϵ)
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
  args...;
  kwargs...) where T

  ZeroRBAffineDecomposition()
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
      [entire_bs_row'*col for col in eachcol(entire_bs_component)]
    end

    function compress_space(
      bs_component::NnzArray,
      rbspace_row::$Trb{<:AbstractMatrix},
      rbspace_col::$Trb{<:AbstractMatrix})

      bs_row = get_basis_space(rbspace_row)
      bs_col = get_basis_space(rbspace_col)
      entire_bs_row = recast(bs_row)
      entire_bs_col = recast(bs_col)
      nbasis = size(bs_component.nonzero_val,2)
      [entire_bs_row'*recast(bs_component,n)*entire_bs_col for n in 1:nbasis]
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

  btbt = allocate_matrix(bt_component,time_ndofs,nt_row*nt_col)
  btbt_shift = allocate_matrix(bt_component,time_ndofs-1,nt_row*nt_col)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    btbt[:,(jt-1)*nt_row+it] .= bt_row[:,it].*bt_col[:,jt]
    btbt_shift[:,(jt-1)*nt_row+it] .= bt_row[2:time_ndofs,it].*bt_col[1:time_ndofs-1,jt]
  end

  bt_component.nonzero_val,btbt,btbt_shift
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
