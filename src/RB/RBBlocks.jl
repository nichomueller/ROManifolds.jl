abstract type RBBlock{T,N} end

Base.getindex(b::RBBlock,i...) = b.blocks[i...]
Base.iterate(b::RBBlock,args...) = iterate(b.blocks,args...)
Base.enumerate(b::RBBlock) = enumerate(b.blocks)
Base.axes(b::RBBlock,i...) = axes(b.blocks,i...)
get_blocks(b) = b.blocks
get_nblocks(b) = length(b.blocks)

struct BlockSnapshots{T} <: RBBlock{T,1}
  blocks::Vector{Snapshots{T}}

  function BlockSnapshots(blocks::Vector{Snapshots{T}}) where T
    new{T}(blocks)
  end

  function BlockSnapshots(v::Vector{<:Vector{<:PTArray{T}}}) where T
    nblocks = length(testitem(v))
    blocks = Vector{Snapshots{T}}(undef,nblocks)
    @inbounds for n in 1:nblocks
      vn = map(x->getindex(x,n),v)
      blocks[n] = Snapshots(vn)
    end
    BlockSnapshots(blocks)
  end
end

function Base.getindex(s::BlockSnapshots{T},idx::UnitRange{Int}) where T
  nblocks = get_nblocks(s)
  map(1:nblocks) do row
    srow = s[row]
    srow[idx]
  end
end

function save(info::RBInfo,s::BlockSnapshots)
  if info.save_solutions
    path = joinpath(info.fe_path,"fesnaps")
    save(path,s)
  end
end

function load(info::RBInfo,T::Type{BlockSnapshots})
  path = joinpath(info.fe_path,"fesnaps")
  load(path,T)
end

function recenter(
  fesolver::PThetaMethod,
  s::BlockSnapshots{T},
  μ::Table) where T

  θ = fesolver.θ
  uh0 = fesolver.uh0(μ)
  u0 = get_free_dof_values(uh0)
  nblocks = get_nblocks(rbspace)
  pend = 1
  sθ = map(1:nblocks) do row
    s_row = s[row]
    s1_row = testitem(testitem(s_row.snaps))
    pini = pend
    pend = pini + size(s1_row,1) - 1
    u0_row = map(x->getindex(x,pini:pend),u0)
    s_row.snaps.*θ + [u0_row,s_row.snaps[2:end]...].*(1-θ)
  end
  BlockSnapshots(Snapshots.(sθ))
end

struct BlockNnzMatrix{T} <: RBBlock{T,1}
  blocks::Vector{NnzMatrix{T}}

  function BlockNnzMatrix(blocks::Vector{NnzMatrix{T}}) where T
    @check all([length(nzm) == length(blocks[1]) for nzm in blocks[2:end]])
    new{T}(blocks)
  end
end

function NnzArray(s::BlockSnapshots{T}) where T
  blocks = map(s.snaps) do val
    array = get_array(hcat(val...))
    NnzMatrix(array...)
  end
  BlockNnzMatrix(blocks)
end

struct BlockRBSpace{T} <: RBBlock{T,1}
  blocks::Vector{RBSpace{T}}

  function BlockRBSpace(blocks::Vector{RBSpace{T}}) where T
    new{T}(blocks)
  end

  function BlockRBSpace(bases_space::Vector{Matrix{T}},bases_time::Vector{Matrix{T}}) where T
    blocks = map(RBSpace,bases_space,bases_time)
    BlockRBSpace(blocks)
  end
end

function Base.show(io::IO,rb::BlockRBSpace)
  for (row,block) in enumerate(rb)
    nbs = size(block.basis_space,2)
    nbt = size(block.basis_time,2)
    print(io,"\n")
    printstyled("RB SPACE INFO FIELD $row\n";underline=true)
    print(io,"Reduced basis space with #(basis space, basis time) = ($nbs,$nbt)\n")
  end
end

function save(info::RBInfo,rb::BlockRBSpace)
  if info.save_structures
    path = joinpath(info.rb_path,"rb")
    save(path,rb)
  end
end

function load(info::RBInfo,T::Type{BlockRBSpace})
  path = joinpath(info.rb_path,"rb")
  load(path,T)
end

function num_rb_dofs(rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  ndofs = 0
  @inbounds for i = 1:nblocks
    ndofs += num_rb_dofs(rb[i])
  end
  ndofs
end

function reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  args...;
  kwargs...)

  energy_norm = info.energy_norm
  nblocks = get_nblocks(snaps)
  blocks = map(1:nblocks) do col
    feop_row_col = feop[1,col]
    snaps_col = sols[col]
    energy_norm_col = energy_norm[col]
    norm_matrix = get_norm_matrix(feop,energy_norm_col)
    basis_space_nnz,basis_time = compress(info,feop_row_col,snaps_col,norm_matrix,args...)
    basis_space = recast(basis_space_nnz)
    basis_space,basis_time,norm_matrix
  end
  bases_space = getindex.(blocks,1)
  bases_time = getindex.(blocks,2)
  norm_matrix = getindex.(blocks,3)
  if info.compute_supremizers
    bases_space = add_space_supremizers(bases_space,feop,norm_matrix,args...)
    bases_time = add_time_supremizers(bases_time;kwargs...)
  end
  rbspace = BlockRBSpace(bases_space,bases_time)
  show(rbspace)
  return rbspace
end

function add_space_supremizers(
  bases_space::Vector{<:Matrix},
  feop::PTFEOperator,
  norm_matrix::AbstractVector,
  args...)

  bs_primal,bs_dual... = bases_space
  nm_primal, = norm_matrix
  dual_nfields = length(bs_dual)
  for col in 1:dual_nfields
    println("Computing supremizers in space for dual field $col")
    feop_row_col = feop[1,col+1]
    supr_col = space_supremizers(bs_dual[col],feop_row_col,args...)
    gram_schmidt!(supr_col,bs_primal,nm_primal)
    bs_primal = hcat(bs_primal,supr_col)
  end
  return [bs_primal,bs_dual...]
end

function space_supremizers(
  basis_space::Matrix,
  feop::PTFEOperator,
  params::Table)

  μ = testitem(params)
  u = zero(feop.test)
  t = 0.
  j(du,dv) = integrate(feop.jacs[1](μ,t,u,du,dv),DomainContribution())
  trial_dual = get_trial(feop)
  constraint_mat = assemble_matrix(j,trial_dual(μ,t),feop.test)
  constraint_mat*basis_space
end

function add_time_supremizers(bases_time::Vector{<:Matrix};kwargs...)
  bt_primal,bt_dual... = bases_time
  dual_nfields = length(bt_dual)
  for col in 1:dual_nfields
    println("Computing supremizers in time for dual field $col")
    supr_col = add_time_supremizers(bt_primal,bt_dual[col];kwargs...)
    bt_primal = hcat(bt_primal,supr_col)
  end
  return [bt_primal,bt_dual...]
end

function add_time_supremizers(basis_u::Matrix,basis_p::Matrix;ttol=1e-2)
  basis_up = basis_u'*basis_p

  function enrich(
    basis_u::AbstractMatrix,
    basis_up::AbstractMatrix,
    v::AbstractArray)

    vnew = orth_complement(v,basis_u)
    vnew /= norm(vnew)
    hcat(basis_u,vnew),vcat(basis_up,vnew'*basis_p)
  end

  count = 0
  ntp_minus_ntu = size(basis_p,2) - size(basis_u,2)
  if ntp_minus_ntu > 0
    for ntp = 1:ntp_minus_ntu
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
    end
  end

  ntp = 1
  while ntp ≤ size(basis_up,2)
    proj = ntp == 1 ? zeros(size(basis_up[:,1])) : orth_projection(basis_up[:,ntp],basis_up[:,1:ntp-1])
    dist = norm(basis_up[:,1]-proj)
    if dist ≤ ttol
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
      ntp = 0
    else
      basis_up[:,ntp] -= proj
    end
    ntp += 1
  end

  println("Added $count time supremizers")
  basis_u
end

struct BlockRBAlgebraicContribution{T,N} <: RBBlock{T,N}
  blocks::Array{RBAlgebraicContribution{T},N}
  touched::BitArray{N}

  function BlockRBAlgebraicContribution(
    blocks::Array{RBAlgebraicContribution{T},N},
    touched::BitArray{N}) where {T,N}

    new{T,N}(blocks,touched)
  end
end

const VecBlockRBAlgebraicContribution{T} = BlockRBAlgebraicContribution{T,1}
const MatBlockRBAlgebraicContribution{T} = BlockRBAlgebraicContribution{T,2}

Base.getindex(a::BlockRBAlgebraicContribution,idx...) = a.blocks[idx...],a.touched[idx...]
get_nblocks(a::MatBlockRBAlgebraicContribution) = size(a.blocks,2)

function Base.show(io::IO,a::BlockRBAlgebraicContribution{T}) where T
  for row in axes(a,1), col in axes(a,2)
    a_row_col,touched = a[row,col]
    if touched
      print(io,"\n")
      printstyled("RB ALGEBRAIC CONTRIBUTIONS INFO, BLOCK ($row,$col)\n";underline=true)
      for trian in get_domains(a_row_col)
        atrian = a_row_col[trian]
        red_method = get_reduction_method(atrian)
        red_var = get_reduced_variable(atrian)
        nbs = get_space_ndofs(atrian)
        nbt = get_time_ndofs(atrian)
        print(io,"$red_var on a $trian, reduction in $red_method\n")
        print(io,"number basis vectors in (space, time) = ($nbs,$nbt)\n")
      end
    end
  end
end

function Base.show(io::IO,a::Vector{<:BlockRBAlgebraicContribution})
  map(x->show(io,x),a)
end

function save_algebraic_contrib(path::String,a::VecBlockRBAlgebraicContribution{T}) where T
  for row in 1:get_nblocks(a)
    block,touched = a[row]
    rpath = joinpath(path,"block_$row")
    create_dir!(rpath)
    tpath = joinpath(rpath,"touched")
    save_algebraic_contrib(rpath,block)
    save(tpath,touched)
  end
end

function save_algebraic_contrib(path::String,a::MatBlockRBAlgebraicContribution{T}) where T
  create_dir!(path)
  for (row,col) in index_pairs(get_nblocks(a),get_nblocks(a))
    block,touched = a[row,col]
    rcpath = joinpath(path,"block_$(row)_$(col)")
    create_dir!(rcpath)
    tpath = joinpath(rcpath,"touched")
    save_algebraic_contrib(rcpath,block)
    save(tpath,touched)
  end
end

function load_algebraic_contrib(path::String,::Type{VecBlockRBAlgebraicContribution})
  nblocks = num_active_dirs(path)
  result = map(1:nblocks) do row
    rpath = joinpath(path,"block_$row")
    tpath = joinpath(rpath,"touched")
    block = load_algebraic_contrib(rpath,RBAlgebraicContribution)
    touched = load(tpath,Bool)
    block,touched
  end
  blocks = first.(result)
  touched = last.(result)
  return BlockRBAlgebraicContribution(blocks,touched)
end

function load_algebraic_contrib(path::String,::Type{MatBlockRBAlgebraicContribution})
  nblocks = num_active_dirs(path)
  result = map(index_pairs(nblocks,nblocks)) do (row,col)
    rcpath = joinpath(path,"block_$(row)_$(col)")
    tpath = joinpath(rcpath,"touched")
    block = load_algebraic_contrib(rcpath,RBAlgebraicContribution)
    touched = load(tpath,Bool)
    block,touched
  end
  blocks = first.(result)
  touched = last.(result)
  return BlockRBAlgebraicContribution(blocks,touched)
end

function save(info::RBInfo,a::VecBlockRBAlgebraicContribution)
  if info.save_structures
    path = joinpath(info.rb_path,"rb_rhs")
    save_algebraic_contrib(path,a)
  end
end

function load(info::RBInfo,T::Type{VecBlockRBAlgebraicContribution})
  path = joinpath(info.rb_path,"rb_rhs")
  load_algebraic_contrib(path,T)
end

function save(info::RBInfo,a::Vector{MatBlockRBAlgebraicContribution{T}}) where T
  if info.save_structures
    for i = eachindex(a)
      path = joinpath(info.rb_path,"rb_lhs_$i")
      save_algebraic_contrib(path,a[i])
    end
  end
end

function load(info::RBInfo,::Type{Vector{MatBlockRBAlgebraicContribution}})
  T = load(joinpath(joinpath(joinpath(info.rb_path,"rb_lhs_1"),"block_1_1"),"type"),DataType)
  njacs = num_active_dirs(info.rb_path)
  ad_jacs = Vector{MatBlockRBAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    path = joinpath(info.rb_path,"rb_lhs_$i")
    ad_jacs[i] = load_algebraic_contrib(path,MatBlockRBAlgebraicContribution)
  end
  ad_jacs
end

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::BlockRBSpace{T},
  snaps::Vector{<:PTArray},
  μ::Table) where T

  times = get_times(fesolver)
  nblocks = get_nblocks(rbspace)
  @assert length(snaps) == nblocks
  result = map(1:nblocks) do row
    feop_row_col = feop[row,:]
    vsnaps = vcat(snaps...)
    rbspace_row = rbspace[row]
    touched = check_touched_residuals(feop_row_col,vsnaps,μ,times)
    if touched
      ress,trian = collect_residuals_for_trian(fesolver,feop_row_col,vsnaps,μ,times)
      rbres = compress_component(info,feop_row_col,ress,trian,times,rbspace_row)
    else
      rbres = testvalue(RBAlgebraicContribution{T},feop_row_col;vector=true)
    end
    rbres,touched
  end
  blocks = first.(result)
  touched = last.(result)
  ad_res = BlockRBAlgebraicContribution(blocks,touched)
  show(ad_res)
  return ad_res
end

function collect_compress_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::BlockRBSpace{T},
  snaps::Vector{<:PTArray},
  μ::Table) where T

  times = get_times(fesolver)
  θ = fesolver.θ
  nblocks = get_nblocks(rbspace)

  njacs = length(feop.jacs)
  ad_jacs = Vector{BlockRBAlgebraicContribution{T,2}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    result_i = map(index_pairs(nblocks,nblocks)) do (row,col)
      feop_row_col = feop[row,col]
      snaps_col = snaps[col]
      rbspace_row = rbspace[row]
      rbspace_col = rbspace[col]
      touched = check_touched_jacobians(feop_row_col,snaps_col,μ,times;i)
      if touched
        jacs,trian = collect_jacobians_for_trian(fesolver,feop_row_col,snaps_col,μ,times;i)
        rbjac = compress_component(info,feop_row_col,jacs,trian,times,rbspace_row,rbspace_col;combine_projections)
      else
        rbjac = testvalue(RBAlgebraicContribution{T},feop_row_col;vector=false)
      end
      rbjac,touched
    end
    blocks_i = first.(result_i)
    touched_i = last.(result_i)
    ad_jac_i = BlockRBAlgebraicContribution(blocks_i,touched_i)
    show(ad_jac_i)
    ad_jacs[i] = ad_jac_i
  end
  return ad_jacs
end

function check_touched_residuals(
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table,
  times::Vector{<:Real};
  kwargs...)

  ode_op = get_algebraic_operator(feop)
  test = get_test(feop)
  Us, = allocate_cache(ode_op,μ,times)
  uh = EvaluationFunction(Us[1],sols)
  μ1 = testitem(μ)
  t1 = testitem(times)
  uh1 = testitem(uh)
  dxh1 = ()
  for i in 1:get_order(feop)
    dxh1 = (dxh1...,uh1)
  end
  xh1 = TransientCellField(uh1,dxh1)
  dv = get_fe_basis(test)
  int = feop.res(μ1,t1,xh1,dv)
  return !isnothing(int)
end

function check_touched_jacobians(
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table,
  times::Vector{<:Real};
  i=1)

  ode_op = get_algebraic_operator(feop)
  test = get_test(feop)
  trial = get_trial(feop)
  Us, = allocate_cache(ode_op,μ,times)
  uh = EvaluationFunction(Us[1],sols)
  μ1 = testitem(μ)
  t1 = testitem(times)
  uh1 = testitem(uh)
  dxh1 = ()
  for i in 1:get_order(feop)
    dxh1 = (dxh1...,uh1)
  end
  xh1 = TransientCellField(uh1,dxh1)
  dv = get_fe_basis(test)
  du = get_trial_fe_basis(trial(nothing,nothing))
  int = feop.jacs[i](μ1,t1,xh1,du,dv)
  return !isnothing(int)
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbres::BlockRBAlgebraicContribution{T,1},
  rbspace::BlockRBSpace{T},
  sols::Vector{<:PTArray},
  args...) where T

  nblocks = get_nblocks(rbres)
  blocks = map(1:nblocks) do row
    feop_row = feop[row,:]
    vsnaps = vcat(sols...)
    rbres_row,touched_row = rbres[row]
    rbspace_row = rbspace[row]
    if touched_row
      collect_rhs_contributions!(cache,info,feop_row,fesolver,rbres_row,rbspace_row,vsnaps,args...)
    else
      veccache, = cache[1]
      allocate_vector!(veccache,rbspace_row)
    end
  end
  vcat(blocks...)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjacs::Vector{BlockRBAlgebraicContribution{T,2}},
  rbspace::BlockRBSpace{T},
  sols::Vector{<:PTArray},
  args...) where T

  njacs = length(rbjacs)
  nblocks = get_nblocks(testitem(rbjacs))
  rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    blocks = map(index_pairs(nblocks,nblocks)) do (row,col)
      feop_row_col = feop[row,col]
      sols_col = sols[col]
      rb_jac_i_row_col,touched_i_row_col = rb_jac_i[row,col]
      rbspace_row = rbspace[row]
      rbspace_col = rbspace[col]
      if touched_i_row_col
        collect_lhs_contributions!(
            cache,info,feop_row_col,fesolver,rb_jac_i_row_col,rbspace_col,sols_col,args...;i)
      else
        matcache, = cache[1]
        allocate_matrix!(matcache,rbspace_row,rbspace_col)
      end
    end
    rb_jacs_contribs[i] = hvcat(blocks...)
  end
  return sum(rb_jacs_contribs)
end

function save_test(info::RBInfo,snaps::BlockSnapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps_test")
    save(path,snaps)
  end
end

function load_test(info::RBInfo,T::Type{BlockSnapshots})
  path = joinpath(info.fe_path,"fesnaps_test")
  load(path,T)
end

function post_process(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  sol::Vector{<:PTArray},
  params::Table,
  sol_approx::Vector{PTArray{T}},
  stats::NamedTuple) where T

  nblocks = length(sol)
  energy_norm = info.energy_norm
  map(1:nblocks) do col
    feop_col = feop[col,col]
    sol_col = sol[col]
    sol_approx_col = sol_approx[col]
    norm_matrix_col = get_norm_matrix(feop_col,energy_norm[col])
    _sol_col = space_time_matrices(sol_col;nparams=length(params))
    results = RBResults(params,_sol_col,sol_approx_col,stats;name=Symbol("field$col"),norm_matrix_col)
    show(results)
    save(info,results)
    writevtk(info,feop_col,fesolver,results)
  end
  return
end

function allocate_online_cache(
  feop::PTFEOperator,
  fesolver::PODESolver,
  snaps_test::Vector{<:PTArray},
  params::Table)

  vsnaps = vcat(snaps_test...)
  allocate_online_cache(feop,fesolver,vsnaps,params)
end

function initial_guess(
  sols::BlockSnapshots,
  params::Table,
  params_test::Table)

  nblocks = get_nblocks(sols)
  kdtree = KDTree(map(x -> SVector(Tuple(x)),params))
  idx_dist = map(x -> nn(kdtree,SVector(Tuple(x))),params_test)
  map(1:nblocks) do row
    srow = sols[row]
    srow[first.(idx_dist)]
  end
end
