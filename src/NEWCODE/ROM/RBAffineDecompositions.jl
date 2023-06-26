struct RBIntegrationDomain
  meas::Measure
  idx::Vector{Int}

  function RBIntegrationDomain(
    component::SingleFieldSnapshots,
    trian::Triangulation,
    interp_idx::Vector{Int})

    degree = Gridap.FESpaces.get_order(first(trian.grid.reffes))
    nonzero_idx = get_nonzero_idx(component)
    nrows = get_nrows(component)
    entire_interp_idx = nonzero_idx[interp_idx]
    entire_interp_rows,_ = from_vec_to_mat_idx(entire_interp_idx,nrows)
    red_integr_cells = find_cells(entire_interp_rows,trian)
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
    interp_idx_time::Vector{Int};
    st_mdeim=true)

    degree = Gridap.FESpaces.get_order(first(trian.grid.reffes))
    nonzero_idx = get_nonzero_idx(component)
    nrows = get_nrows(component)
    entire_interp_idx_space = nonzero_idx[interp_idx_space]
    entire_interp_rows_space,_ = from_vec_to_mat_idx(entire_interp_idx_space,nrows)
    red_integr_cells = find_cells(entire_interp_rows_space,trian)
    red_trian = view(trian,red_integr_cells)
    red_meas = Measure(red_trian,degree)
    red_times = st_mdeim ? times[interp_idx_time] : times
    new(red_meas,red_times,entire_interp_idx_space)
  end
end

abstract type RBAffineDecompositions end

struct RBAffineDecomposition <: RBAffineDecompositions
  basis_space::AbstractMatrix
  mdeim_interpolation::LU
  integration_domain::RBIntegrationDomain
end

struct TransientRBAffineDecomposition <: RBAffineDecompositions
  basis_space::AbstractMatrix
  basis_time::Tuple{Vararg{AbstractMatrix}}
  mdeim_interpolation::LU
  integration_domain::TransientRBIntegrationDomain
end

for (Top,Tsol,Tsps,Tspm) in zip(
  (:ParamFEOperator,:ParamTransientFEOperator),
  (:FESolver,:ODESolver),
  (:SingleFieldRBSpace,:TransientSingleFieldRBSpace),
  (:MultiFieldRBSpace,:TransientMultiFieldRBSpace))

  @eval begin
    function compress_jacobian(
      info::RBInfo,
      feop::$Top,
      solver::$Tsol,
      args...;
      kwargs...)

      snaps = load(Snapshots,info)
      params = load(Table,info)
      trians = _collect_trian_jac(feop)
      cjac = RBAlgebraicContribution()
      for trian in trians
        ad_jac = compress_jacobian(feop,solver,trian,snaps,params,args...;
          nsnaps=info.nsnaps_mdeim,kwargs...)
        add_contribution!(cjac,trian,ad_jac)
      end
      cjac
    end

    function compress_jacobian(
      feop::$Top,
      solver::$Tsol,
      trian::Triangulation,
      s::MultiFieldSnapshots,
      params::Table,
      rbspace::$Tspm;
      kwargs...)

      nfields = get_nfields(s)
      lazy_map(1:nfields) do row
        lazy_map(1:nfields) do col
          compress_jacobian(feop,solver,trian,
            (rbspace[row],rbspace[col]),s[col],params,(row,col);kwargs...)
        end
      end
    end

    function compress_jacobian(
      feop::$Top,
      solver::$Tsol,
      trian::Triangulation,
      s::SingleFieldSnapshots,
      params::Table,
      rbspace::$Tsps,
      filter=(1,1);
      kwargs...)

      compress_jacobian(feop,solver,trian,
        (rbspace,rbspace),s,params,filter;kwargs...)
    end
  end

end

function compress_jacobian(
  feop::ParamTransientFEOperator,
  solver::ODESolver,
  trian::Triangulation,
  rbspace::NTuple{2,TransientSingleFieldRBSpace},
  s::SingleFieldSnapshots,
  params::Table,
  filter::Tuple{Vararg{Int}};
  kwargs...)

  j = assemble_jacobian(feop,solver,trian,s,params,filter)
  compress_component(j,solver,trian,rbspace...;kwargs...)
end

function compress_component(
  component::SingleFieldSnapshots,
  solver::FESolver,
  trian::Triangulation,
  args...;
  kwargs...)

  rbspace_component = tpod(component;kwargs...)
  interp_idx = get_interpolation_idx(rbspace_component)
  integr_domain = RBIntegrationDomain(component,trian,interp_idx)

  bs = get_basis_space(rbspace_component)
  interp_bs = bs[interp_idx,:]
  lu_interp_bs = lu(interp_bs)

  proj_bs = compress(solver,rbspace_component,args...)

  RBAffineDecomposition(proj_bs,lu_interp_bs,integr_domain)
end

function compress_component(
  solver::ODESolver,
  component::SingleFieldSnapshots,
  trian::Triangulation,
  args...;
  st_mdeim=true,
  kwargs...)

  times = get_times(solver)

  bs,bt = transient_tpod(component;kwargs...)
  interp_idx_space,interp_idx_time = get_interpolation_idx(bs,bt)
  integr_domain = TransientRBIntegrationDomain(
    component,trian,times,interp_idx_space,interp_idx_time;st_mdeim)

  interp_bs = bs[interp_idx_space,:]
  interp_bt = bt[interp_idx_time,:]
  interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
  lu_interp_bst = lu(interp_bst)

  proj_bs,proj_bt... = compress(solver,bs,bt,args...)

  TransientRBAffineDecomposition(proj_bs,proj_bt,lu_interp_bst,integr_domain)
end

function get_interpolation_idx(basis_space::NnzArray)
  idx_space = get_interpolation_idx(basis_space.array)
  idx_space
end

function get_interpolation_idx(basis_space::NnzArray,basis_time::NnzArray)
  idx_space = get_interpolation_idx(basis_space.array)
  idx_time = get_interpolation_idx(basis_time.array)
  idx_space,idx_time
end

function get_interpolation_idx(mat::AbstractMatrix)
  n = size(mat,2)
  idx = zeros(Int,size(mat,2))
  idx[1] = argmax(abs.(mat[:,1]))
  @inbounds for i = 2:n
    res = mat[:,i] - mat[:,1:i-1]*(mat[idx[1:i-1],1:i-1] \ mat[idx[1:i-1],i])
    idx[i] = argmax(abs.(res))
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
  solver::ODESolver,
  bs_component::NnzArray,
  bt_component::NnzArray,
  args...)

  compress_space(bs_component,args...),compress_time(solver,bt_component,args...)
end

for Top in (:SingleFieldRBSpace,:TransientSingleFieldRBSpace)

  @eval begin
    function compress_space(
      bs_component::NnzArray,
      rbspace_row::$Top{<:AbstractMatrix})

      bs = get_basis_space(rbspace_row)
      entire_bs_component = recast(bs_component)
      entire_bs = recast(bs)
      entire_bs'*entire_bs_component
    end

    function compress_space(
      bs_component::NnzArray,
      rbspace_row::$Top{<:AbstractMatrix},
      rbspace_col::$Top{<:AbstractMatrix})

      bs_row = get_basis_space(rbspace_row)
      bs_col = get_basis_space(rbspace_col)
      entire_bs_row = recast(bs_row)
      entire_bs_col = recast(bs_col)
      proj_blocks = [reshape(entire_bs_row'*recast(bs_component,ncol)*entire_bs_col,:)
        for ncol in axes(bs_component,2)]
      hcat(proj_blocks...)
    end
  end
end

function compress_time(
  ::θMethod,
  bt_component::NnzArray,
  rbspace_row::TransientSingleFieldRBSpace{<:AbstractMatrix})

  bt = get_basis_time(rbspace_row)
  bt_component,bt
end

function compress_time(
  ::θMethod,
  bt_component::NnzArray,
  rbspace_row::TransientSingleFieldRBSpace{<:AbstractMatrix},
  rbspace_col::TransientSingleFieldRBSpace{<:AbstractMatrix})

  bt_row = get_basis_time(rbspace_row)
  bt_col = get_basis_time(rbspace_col)
  time_ndofs = size(bt_component,1)
  nt_row,nt_col = size(bt_row,2),size(bt_col,2)

  btbt = allocate_matrix(bt_component,time_ndofs,nrow*ncol)
  btbt_shift = allocate_matrix(bt_component,time_ndofs-1,nrow*ncol)
  @inbounds for jt = 1:nt_col, it = 1:nt_row
    btbt[:,(jt-1)*nt_row+it] .= bt_row[:,it].*bt_col[:,jt]
    btbt_shift[:,(jt-1)*nt_row+it] .= bt_row[2:time_ndofs,it].*bt_col[1:time_ndofs-1,jt]
  end

  bt_component,btbt,btbt_shift
end

function find_cells(idx::Vector{Int},trian::Triangulation)
  cell_dof_ids = get_cell_dof_ids(trian)
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
