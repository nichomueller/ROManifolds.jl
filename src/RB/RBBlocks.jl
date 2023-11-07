abstract type RBBlock{T,N} end

Base.getindex(b::RBBlock,i::Int...) = b.blocks[i...]
Base.iterate(b::RBBlock,args...) = iterate(b.blocks,args...)
Base.enumerate(b::RBBlock) = enumerate(b.blocks)
Base.axes(b::RBBlock,i::Int...) = axes(b.blocks,i...)
Base.lastindex(b::RBBlock) = lastindex(testitem(b))
Arrays.testitem(b::RBBlock) = testitem(b.blocks)
get_blocks(b) = b.blocks
get_nblocks(b) = length(b.blocks)

struct BlockSnapshots{T} <: RBBlock{T,1}
  blocks::Vector{Snapshots{T}}

  function BlockSnapshots(blocks::Vector{Snapshots{T}}) where T
    new{T}(blocks)
  end

  function BlockSnapshots(v::Vector{Vector{NonaffinePTArray{T}}}) where T
    nblocks = length(testitem(v))
    blocks = Vector{Snapshots{T}}(undef,nblocks)
    @inbounds for n in 1:nblocks
      vn = map(x->getindex(x,n),v)
      blocks[n] = Snapshots(vn)
    end
    BlockSnapshots(blocks)
  end
end

function Base.getindex(s::BlockSnapshots,idx::UnitRange{Int})
  nblocks = get_nblocks(s)
  blocks = map(1:nblocks) do row
    srow = s[row]
    srow[idx]
  end
  vcat(blocks...)
end

function recenter(s::BlockSnapshots,uh0::PTFEFunction;θ::Real=1)
  nblocks = get_nblocks(s)
  sθ = map(1:nblocks) do row
    recenter(s[row],uh0[row];θ)
  end
  BlockSnapshots(sθ)
end

function save(info::RBInfo,s::BlockSnapshots)
  path = joinpath(info.fe_path,"fesnaps")
  save(path,s)
end

function load(info::RBInfo,T::Type{BlockSnapshots})
  path = joinpath(info.fe_path,"fesnaps")
  load(path,T)
end

struct BlockNnzMatrix{T} <: RBBlock{T,1}
  blocks::Vector{NnzMatrix{T}}

  function BlockNnzMatrix(blocks::Vector{NnzMatrix{T}}) where T
    @check all([length(nzm) == length(blocks[1]) for nzm in blocks[2:end]])
    new{T}(blocks)
  end
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

function field_offsets(rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  offsets = zeros(Int,nblocks+1)
  @inbounds for block = 1:nblocks
    ndofs = get_rb_ndofs(rb[block])
    offsets[block+1] = offsets[block] + ndofs
  end
  offsets
end

function save(info::RBInfo,rb::BlockRBSpace)
  path = joinpath(info.rb_path,"rb")
  save(path,rb)
end

function load(info::RBInfo,T::Type{BlockRBSpace})
  path = joinpath(info.rb_path,"rb")
  load(path,T)
end

function get_rb_ndofs(rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  ndofs = 0
  @inbounds for i = 1:nblocks
    ndofs += get_rb_ndofs(rb[i])
  end
  ndofs
end

function recast(x::PTArray,rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  offsets = field_offsets(rb)
  blocks = map(1:nblocks) do row
    rb_row = rb[row]
    x_row = get_at_offsets(x,offsets,row)
    recast(x_row,rb_row)
  end
  return vcat(blocks...)
end

function space_time_projection(x::PTArray,op::PTAlgebraicOperator,rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  offsets = field_offsets(op.odeop.feop.test)
  blocks = map(1:nblocks) do row
    rb_row = rb[row]
    x_row = get_at_offsets(x,offsets,row)
    space_time_projection(x_row,rb_row)
  end
  return vcat(blocks...)
end

function reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::BlockSnapshots)

  println("Computing RB space")

  ϵ = info.ϵ
  nsnaps_state = info.nsnaps_state
  norm_style = info.norm_style
  nblocks = get_nblocks(snaps)
  blocks = map(1:nblocks) do col
    snaps_col = snaps[col]
    norm_matrix = get_norm_matrix(info,feop,norm_style[col])
    reduced_basis(snaps_col,norm_matrix;ϵ,nsnaps_state)
  end
  if info.compute_supremizers
    bases_space = add_space_supremizers(info,feop,blocks)
    bases_time = add_time_supremizers(blocks)
  end
  return BlockRBSpace(bases_space,bases_time)
end

function add_space_supremizers(
  info::RBInfo,
  feop::PTFEOperator,
  blocks::Vector{RBSpace{T}}) where T

  bs_primal,bs_dual... = map(get_basis_space,blocks)
  primal_norm_style = first(info.norm_style)
  nm_primal = get_norm_matrix(info,feop,primal_norm_style)
  dual_nfields = length(bs_dual)
  for col in 1:dual_nfields
    println("Computing supremizers in space for dual field $col")
    feop_row_col = feop[1,col+1]
    supr_col = space_supremizers(bs_dual[col],feop_row_col)
    gram_schmidt!(supr_col,bs_primal,nm_primal)
    bs_primal = hcat(bs_primal,supr_col)
  end
  return [bs_primal,bs_dual...]
end

function space_supremizers(basis_space::Matrix,feop::PTFEOperator)
  μ = realization(feop)
  u = zero(feop.test)
  t = 0.
  jac = get_jacobian(feop)
  j(du,dv) = integrate(jac[1](μ,t,u,du,dv))
  trial_dual = get_trial(feop)
  constraint_mat = assemble_matrix(j,trial_dual(μ,t),feop.test)
  constraint_mat*basis_space
end

function add_time_supremizers(blocks::Vector{<:RBSpace})
  bt_primal,bt_dual... = map(get_basis_time,blocks)
  dual_nfields = length(bt_dual)
  for col in 1:dual_nfields
    println("Computing supremizers in time for dual field $col")
    bt_primal = add_time_supremizers(bt_primal,bt_dual[col])
  end
  return [bt_primal,bt_dual...]
end

function add_time_supremizers(basis_u::Matrix,basis_p::Matrix;ttol=1e-2)
  basis_up = basis_u'*basis_p

  function enrich(
    basis_u::AbstractMatrix,
    basis_up::AbstractMatrix,
    v::AbstractArray)

    vnew = copy(v)
    orth_complement!(vnew,basis_u)
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

  if count > 0
    println("Added $count time supremizer(s)")
  end
  basis_u
end

function get_ptoperator(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  rbspace::BlockRBSpace{T},
  params::Table;
  kwargs...) where T

  nblocks = get_nblocks(rbspace)
  space_ndofs = cumsum([0,map(get_space_ndofs,rbspace.blocks)...])
  rb_space_ndofs = map(get_rb_space_ndofs,rbspace.blocks)
  basis_space = zeros(T,last(space_ndofs),maximum(rb_space_ndofs))
  @inbounds for n = 1:nblocks
    basis_space[space_ndofs[n]+1:space_ndofs[n+1],1:rb_space_ndofs[n]] = get_basis_space(rbspace[n])
  end
  basis_time = get_basis_time(testitem(rbspace))
  _rbspace = RBSpace(basis_space,basis_time)
  get_ptoperator(fesolver,feop,_rbspace,params;kwargs...)
end

abstract type BlockRBAlgebraicContribution{T,N} <: RBBlock{T,N} end

struct BlockRBVecAlgebraicContribution{T} <: BlockRBAlgebraicContribution{T,1}
  blocks::Vector{RBVecAlgebraicContribution{T}}
  touched::Vector{Bool}

  function BlockRBVecAlgebraicContribution(
    blocks::Vector{RBVecAlgebraicContribution{T}},
    touched::Vector{Bool}) where T

    new{T}(blocks,touched)
  end
end

struct BlockRBMatAlgebraicContribution{T} <: BlockRBAlgebraicContribution{T,2}
  blocks::Matrix{RBMatAlgebraicContribution{T}}
  touched::Matrix{Bool}

  function BlockRBMatAlgebraicContribution(
    blocks::Matrix{RBMatAlgebraicContribution{T}},
    touched::Matrix{Bool}) where T

    new{T}(blocks,touched)
  end
end

get_nblocks(a::BlockRBMatAlgebraicContribution) = size(a.blocks,2)

function save_algebraic_contrib(path::String,a::BlockRBVecAlgebraicContribution)
  tpath = joinpath(path,"touched")
  create_dir!(tpath)
  save(tpath,a.touched)
  for row in 1:get_nblocks(a)
    if a.touched[row]
      rpath = joinpath(path,"block_$row")
      create_dir!(rpath)
      save_algebraic_contrib(rpath,a.blocks[row])
    end
  end
end

function save_algebraic_contrib(path::String,a::BlockRBMatAlgebraicContribution)
  tpath = joinpath(path,"touched")
  create_dir!(tpath)
  save(tpath,a.touched)
  for (row,col) in index_pairs(get_nblocks(a),get_nblocks(a))
    if a.touched[row,col]
      rcpath = joinpath(path,"block_$(row)_$(col)")
      create_dir!(rcpath)
      save_algebraic_contrib(rcpath,a.blocks[row,col])
    end
  end
end

function load_algebraic_contrib(path::String,::Type{BlockRBVecAlgebraicContribution{T}}) where T
  tpath = joinpath(path,"touched")
  touched = load(tpath,Vector{Bool})
  nblocks = length(touched)
  blocks = Vector{RBVecAlgebraicContribution{T}}(undef,nblocks)
  for row = 1:nblocks
    if touched[row]
      rpath = joinpath(path,"block_$row")
      blocks[row] = load_algebraic_contrib(rpath,RBVecAlgebraicContribution{T})
    end
  end
  return BlockRBVecAlgebraicContribution(blocks,touched)
end

function load_algebraic_contrib(path::String,::Type{BlockRBMatAlgebraicContribution{T}}) where T
  tpath = joinpath(path,"touched")
  touched = load(tpath,Matrix{Bool})
  nblocks = size(touched,1)
  blocks = Matrix{RBMatAlgebraicContribution{T}}(undef,nblocks,nblocks)
  for (row,col) = index_pairs(nblocks,nblocks)
    if touched[row,col]
      rcpath = joinpath(path,"block_$(row)_$(col)")
      blocks[row,col] = load_algebraic_contrib(rcpath,RBMatAlgebraicContribution{T})
    end
  end
  return BlockRBMatAlgebraicContribution(blocks,touched)
end

function save(info::RBInfo,a::BlockRBVecAlgebraicContribution)
  path = joinpath(info.rb_path,"rb_rhs")
  save_algebraic_contrib(path,a)
end

function load(info::RBInfo,::Type{BlockRBVecAlgebraicContribution{T}}) where T
  path = joinpath(info.rb_path,"rb_rhs")
  load_algebraic_contrib(path,BlockRBVecAlgebraicContribution{T})
end

function save(info::RBInfo,a::Vector{BlockRBMatAlgebraicContribution})
  for i = eachindex(a)
    path = joinpath(info.rb_path,"rb_lhs_$i")
    save_algebraic_contrib(path,a[i])
  end
end

function load(info::RBInfo,::Type{Vector{BlockRBMatAlgebraicContribution{T}}}) where T
  njacs = num_active_dirs(info.rb_path)
  ad_jacs = Vector{BlockRBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    path = joinpath(info.rb_path,"rb_lhs_$i")
    ad_jacs[i] = load_algebraic_contrib(path,BlockRBMatAlgebraicContribution{T})
  end
  ad_jacs
end

function save(info::RBInfo,a::NTuple{2,BlockRBVecAlgebraicContribution})
  a_lin,a_nlin = a
  path_lin = joinpath(info.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(info.rb_path,"rb_rhs_nlin")
  save_algebraic_contrib(path_lin,a_lin)
  save_algebraic_contrib(path_nlin,a_nlin)
end

function load(info::RBInfo,::Type{NTuple{2,BlockRBVecAlgebraicContribution{T}}}) where T
  path_lin = joinpath(info.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(info.rb_path,"rb_rhs_nlin")
  a_lin = load_algebraic_contrib(path_lin,BlockRBVecAlgebraicContribution{T})
  a_nlin = load_algebraic_contrib(path_nlin,BlockRBVecAlgebraicContribution{T})
  a_lin,a_nlin
end

function save(info::RBInfo,a::NTuple{3,Vector{<:BlockRBMatAlgebraicContribution}})
  a_lin,a_nlin,a_aux = a
  for i = eachindex(a_lin)
    path_lin = joinpath(info.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(info.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(info.rb_path,"rb_lhs_aux_$i")
    save_algebraic_contrib(path_lin,a_lin[i])
    save_algebraic_contrib(path_nlin,a_nlin[i])
    save_algebraic_contrib(path_aux,a_aux[i])
  end
end

function load(info::RBInfo,::Type{NTuple{3,Vector{BlockRBMatAlgebraicContribution{T}}}}) where T
  njacs = num_active_dirs(info.rb_path)
  ad_jacs_lin = Vector{BlockRBMatAlgebraicContribution{T}}(undef,njacs)
  ad_jacs_nlin = Vector{BlockRBMatAlgebraicContribution{T}}(undef,njacs)
  ad_jacs_aux = Vector{BlockRBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    path_lin = joinpath(info.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(info.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(info.rb_path,"rb_lhs_aux_$i")
    ad_jacs_lin[i] = load_algebraic_contrib(path_lin,BlockRBMatAlgebraicContribution{T})
    ad_jacs_nlin[i] = load_algebraic_contrib(path_nlin,BlockRBMatAlgebraicContribution{T})
    ad_jacs_aux[i] = load_algebraic_contrib(path_aux,BlockRBMatAlgebraicContribution{T})
  end
  ad_jacs_lin,ad_jacs_nlin,ad_jacs_aux
end

function collect_compress_rhs(
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbspace::BlockRBSpace{T}) where T

  nblocks = get_nblocks(rbspace)
  blocks = Vector{RBVecAlgebraicContribution{T}}(undef,nblocks)
  touched = Vector{Bool}(undef,nblocks)
  for row = 1:nblocks
    op_row_col = op[row,:]
    touched[row] = check_touched_residuals(op_row_col)
    if touched[row]
      rbspace_row = rbspace[row]
      ress,trian = collect_residuals_for_trian(op_row_col)
      ad_res = RBVecAlgebraicContribution(T)
      compress_component!(ad_res,info,op_row_col,ress,trian,rbspace_row)
      blocks[row] = ad_res
    end
  end

  BlockRBVecAlgebraicContribution(blocks,touched)
end

function collect_compress_lhs(
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbspace::BlockRBSpace{T};
  θ::Real=1) where T

  nblocks = get_nblocks(rbspace)
  njacs = length(op.odeop.feop.jacs)
  ad_jacs = Vector{BlockRBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
    touched_i = Matrix{Bool}(undef,nblocks,nblocks)
    blocks_i = Matrix{RBMatAlgebraicContribution{T}}(undef,nblocks,nblocks)
    for (row,col) = index_pairs(nblocks,nblocks)
      op_row_col = op[row,col]
      touched_i[row,col] = check_touched_jacobians(op_row_col;i)
      if touched_i[row,col]
        rbspace_row = rbspace[row]
        rbspace_col = rbspace[col]
        jacs,trian = collect_jacobians_for_trian(op_row_col;i)
        ad_jac = RBMatAlgebraicContribution(T)
        compress_component!(
          ad_jac,info,op_row_col,jacs,trian,rbspace_row,rbspace_col;combine_projections)
        blocks_i[row,col] = ad_jac
      end
    end
    ad_jacs[i] = BlockRBMatAlgebraicContribution(blocks_i,touched_i)
  end

  return ad_jacs
end

function check_touched_residuals(op::PTAlgebraicOperator)
  feop = op.odeop.feop
  test = get_test(feop)
  Us, = op.ode_cache
  uh = EvaluationFunction(Us[1],op.u0)
  μ1 = testitem(op.μ)
  t1 = testitem(op.tθ)
  uh1 = testitem(uh)
  dxh1 = ()
  for i in 1:get_order(feop)
    dxh1 = (dxh1...,uh1)
  end
  xh1 = TransientCellField(uh1,dxh1)
  dv = get_fe_basis(test)
  res = get_residual(feop)
  int = res(μ1,t1,xh1,dv)
  return !isnothing(int)
end

function check_touched_jacobians(op::PTAlgebraicOperator;i=1)
  feop = op.odeop.feop
  test = get_test(feop)
  trial = get_trial(feop)
  Us, = op.ode_cache
  uh = EvaluationFunction(Us[1],op.u0)
  μ1 = testitem(op.μ)
  t1 = testitem(op.tθ)
  uh1 = testitem(uh)
  dxh1 = ()
  for i in 1:get_order(feop)
    dxh1 = (dxh1...,uh1)
  end
  xh1 = TransientCellField(uh1,dxh1)
  dv = get_fe_basis(test)
  du = get_trial_fe_basis(trial(nothing,nothing))
  jac = get_jacobian(feop)
  int = jac[i](μ1,t1,xh1,du,dv)
  return !isnothing(int)
end

function collect_rhs_contributions!(
  cache,
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbres::BlockRBVecAlgebraicContribution{T},
  rbspace::BlockRBSpace{T}) where T

  nblocks = get_nblocks(rbres)
  blocks = Vector{PTArray{Vector{T}}}(undef,nblocks)
  for row = 1:nblocks
    op_row_col = op[row,:]
    rbspace_row = rbspace[row]
    cache_row = cache_at_index(cache,op,row)
    if rbres.touched[row]
      blocks[row] = collect_rhs_contributions!(
        cache_row,info,op_row_col,rbres[row],rbspace_row)
    else
      nrow = get_rb_ndofs(rbspace_row)
      blocks[row] = AffinePTArray(zeros(T,nrow),length(op.μ))
    end
  end
  vcat(blocks...)
end

function collect_lhs_contributions!(
  cache,
  info::RBInfo,
  op::PTAlgebraicOperator,
  rbjacs::Vector{BlockRBMatAlgebraicContribution{T}},
  rbspace::BlockRBSpace{T}) where T

  njacs = length(rbjacs)
  nblocks = get_nblocks(testitem(rbjacs))
  rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    blocks = Matrix{PTArray{Matrix{T}}}(undef,nblocks,nblocks)
    for (row,col) = index_pairs(nblocks,nblocks)
      op_row_col = op[row,col]
      rbspace_row = rbspace[row]
      rbspace_col = rbspace[col]
      cache_row_col = cache_at_index(cache,op,row,col)
      if rb_jac_i.touched[row,col]
        blocks[row,col] = collect_lhs_contributions!(
          cache_row_col,info,op_row_col,rb_jac_i[row,col],rbspace_row,rbspace_col;i)
      else
        nrow = get_rb_ndofs(rbspace_row)
        ncol = get_rb_ndofs(rbspace_col)
        blocks[row,col] = AffinePTArray(zeros(T,nrow,ncol),length(op.μ))
      end
    end
    rb_jacs_contribs[i] = hvcat(nblocks,blocks...)
  end
  return rb_jacs_contribs
end

function cache_at_index(cache,op::PTAlgebraicOperator,row::Int)
  coeff_cache,rb_cache = cache
  b,solve_cache... = coeff_cache
  offsets = field_offsets(op.odeop.feop.test)
  b_idx = get_at_offsets(b,offsets,row)
  return (b_idx,solve_cache...),rb_cache
end

function cache_at_index(cache,op::PTAlgebraicOperator,row::Int,col::Int)
  coeff_cache,rb_cache = cache
  A,solve_cache... = coeff_cache
  offsets = field_offsets(op.odeop.feop.test)
  A_idx = get_at_offsets(A,offsets,row,col)
  return (A_idx,solve_cache...),rb_cache
end
