abstract type RBBlock{T,N} end

Base.eltype(::RBBlock{T,N} where N) where T = T
Base.ndims(::RBBlock{T,N} where T) where N = N
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

function Base.getindex(s::BlockSnapshots,idx::UnitRange{Int})
  nblocks = get_nblocks(s)
  blocks = map(1:nblocks) do row
    srow = s[row]
    srow[idx]
  end
  vcat(blocks...)
end

function recenter(s::BlockSnapshots,uh0::PTFEFunction;kwargs...)
  nblocks = get_nblocks(s)
  sθ = map(1:nblocks) do row
    recenter(s[row],uh0[row];kwargs...)
  end
  BlockSnapshots(sθ)
end

function save(rbinfo::BlockRBInfo,s::BlockSnapshots)
  path = joinpath(rbinfo.fe_path,"fesnaps")
  save(path,s)
end

function load(rbinfo::BlockRBInfo,T::Type{BlockSnapshots{S}}) where S
  path = joinpath(rbinfo.fe_path,"fesnaps")
  load(path,T)
end

struct BlockRBSpace{T} <: RBBlock{T,1}
  blocks::Vector{RBSpace{T}}
end

function BlockRBSpace(bases_space::Vector{Matrix{T}},bases_time::Vector{Matrix{T}}) where T
  blocks = map(RBSpace,bases_space,bases_time)
  BlockRBSpace(blocks)
end

function rb_offsets(rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  offsets = zeros(Int,nblocks+1)
  @inbounds for block = 1:nblocks
    offsets[block+1] = offsets[block] + get_rb_ndofs(rb[block])
  end
  offsets
end

function fe_offsets(rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  offsets = zeros(Int,nblocks+1)
  @inbounds for block = 1:nblocks
    offsets[block+1] = offsets[block] + size(get_basis_space(rb[block]),1)
  end
  offsets
end

function save(rbinfo::BlockRBInfo,rb::BlockRBSpace)
  path = joinpath(rbinfo.rb_path,"rb")
  save(path,rb)
end

function load(rbinfo::BlockRBInfo,T::Type{BlockRBSpace{S}}) where S
  path = joinpath(rbinfo.rb_path,"rb")
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
  offsets = rb_offsets(rb)
  blocks = map(1:nblocks) do row
    rb_row = rb[row]
    x_row = get_at_offsets(x,offsets,row)
    recast(x_row,rb_row)
  end
  return vcat(blocks...)
end

function space_time_projection(x::PTArray,rb::BlockRBSpace)
  nblocks = get_nblocks(rb)
  offsets = fe_offsets(rb)
  blocks = map(1:nblocks) do row
    rb_row = rb[row]
    x_row = get_at_offsets(x,offsets,row)
    space_time_projection(x_row,rb_row)
  end
  return vcat(blocks...)
end

function reduced_basis(
  rbinfo::BlockRBInfo,
  feop::PTFEOperator,
  snaps::BlockSnapshots)

  nblocks = get_nblocks(snaps)
  blocks = map(1:nblocks) do col
    rbinfo_col = rbinfo[col]
    feop_col = feop[col,col]
    snaps_col = snaps[col]
    reduced_basis(rbinfo_col,feop_col,snaps_col)
  end
  if rbinfo.compute_supremizers
    bases_space = add_space_supremizers(rbinfo,feop,blocks)
    bases_time = add_time_supremizers(blocks)
  end
  return BlockRBSpace(bases_space,bases_time)
end

function add_space_supremizers(
  rbinfo::BlockRBInfo,
  feop::PTFEOperator,
  blocks::Vector{RBSpace{T}}) where T

  bs_primal,bs_dual... = map(get_basis_space,blocks)
  nm_primal = get_norm_matrix(rbinfo[1],feop)
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
end

struct BlockRBMatAlgebraicContribution{T} <: BlockRBAlgebraicContribution{T,2}
  blocks::Matrix{RBMatAlgebraicContribution{T}}
  touched::Matrix{Bool}
end

get_nblocks(a::BlockRBMatAlgebraicContribution) = size(a.blocks,2)

function save_algebraic_contrib(path::String,a::BlockRBVecAlgebraicContribution)
  create_dir(path)
  tpath = joinpath(path,"touched")
  save(tpath,a.touched)
  for row in 1:get_nblocks(a)
    if a.touched[row]
      rpath = joinpath(path,"block_$row")
      save_algebraic_contrib(rpath,a.blocks[row])
    end
  end
end

function save_algebraic_contrib(path::String,a::BlockRBMatAlgebraicContribution)
  create_dir(path)
  tpath = joinpath(path,"touched")
  save(tpath,a.touched)
  for (row,col) in index_pairs(get_nblocks(a),get_nblocks(a))
    if a.touched[row,col]
      adpath = joinpath(path,"block_$(row)_$(col)")
      save_algebraic_contrib(adpath,a.blocks[row,col])
    end
  end
end

function load_algebraic_contrib(path::String,::Type{BlockRBVecAlgebraicContribution{T}},args...) where T
  S = RBVecAlgebraicContribution{T}
  tpath = joinpath(path,"touched")
  touched = load(tpath,Vector{Bool})
  nblocks = length(touched)
  blocks = Vector{S}(undef,nblocks)
  for row = 1:nblocks
    if touched[row]
      rpath = joinpath(path,"block_$row")
      blocks[row] = load_algebraic_contrib(rpath,S,args...)
    end
  end
  return BlockRBVecAlgebraicContribution(blocks,touched)
end

function load_algebraic_contrib(path::String,::Type{BlockRBMatAlgebraicContribution{T}},args...) where T
  S = RBMatAlgebraicContribution{T}
  tpath = joinpath(path,"touched")
  touched = load(tpath,Matrix{Bool})
  nblocks = size(touched,1)
  blocks = Matrix{S}(undef,nblocks,nblocks)
  for (row,col) = index_pairs(nblocks,nblocks)
    if touched[row,col]
      adpath = joinpath(path,"block_$(row)_$(col)")
      blocks[row,col] = load_algebraic_contrib(adpath,S,args...)
    end
  end
  return BlockRBMatAlgebraicContribution(blocks,touched)
end

function save(rbinfo::BlockRBInfo,a::BlockRBVecAlgebraicContribution)
  path = joinpath(rbinfo.rb_path,"rb_rhs")
  save_algebraic_contrib(path,a)
end

function load(rbinfo::BlockRBInfo,::Type{BlockRBVecAlgebraicContribution{T}},args...) where T
  S = BlockRBVecAlgebraicContribution{T}
  path = joinpath(rbinfo.rb_path,"rb_rhs")
  load_algebraic_contrib(path,S,args...)
end

function save(rbinfo::BlockRBInfo,a::Vector{<:BlockRBMatAlgebraicContribution})
  for i = eachindex(a)
    path = joinpath(rbinfo.rb_path,"rb_lhs_$i")
    save_algebraic_contrib(path,a[i])
  end
end

function load(rbinfo::BlockRBInfo,::Type{Vector{BlockRBMatAlgebraicContribution{T}}},args...) where T
  S = BlockRBMatAlgebraicContribution{T}
  njacs = num_active_dirs(rbinfo.rb_path)
  ad_jacs = Vector{S}(undef,njacs)
  for i = 1:njacs
    path = joinpath(rbinfo.rb_path,"rb_lhs_$i")
    ad_jacs[i] = load_algebraic_contrib(path,S,args...)
  end
  ad_jacs
end

function save(rbinfo::BlockRBInfo,a::NTuple{2,BlockRBVecAlgebraicContribution})
  a_lin,a_nlin = a
  path_lin = joinpath(rbinfo.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(rbinfo.rb_path,"rb_rhs_nlin")
  save_algebraic_contrib(path_lin,a_lin)
  save_algebraic_contrib(path_nlin,a_nlin)
end

function load(rbinfo::BlockRBInfo,::Type{NTuple{2,BlockRBVecAlgebraicContribution{T}}},args...) where T
  S = BlockRBVecAlgebraicContribution{T}
  path_lin = joinpath(rbinfo.rb_path,"rb_rhs_lin")
  path_nlin = joinpath(rbinfo.rb_path,"rb_rhs_nlin")
  a_lin = load_algebraic_contrib(path_lin,S,args...)
  a_nlin = load_algebraic_contrib(path_nlin,S,args...)
  a_lin,a_nlin
end

function save(rbinfo::BlockRBInfo,a::NTuple{3,Vector{<:BlockRBMatAlgebraicContribution}})
  a_lin,a_nlin,a_aux = a
  for i = eachindex(a_lin)
    path_lin = joinpath(rbinfo.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(rbinfo.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(rbinfo.rb_path,"rb_lhs_aux_$i")
    save_algebraic_contrib(path_lin,a_lin[i])
    save_algebraic_contrib(path_nlin,a_nlin[i])
    save_algebraic_contrib(path_aux,a_aux[i])
  end
end

function load(rbinfo::BlockRBInfo,::Type{NTuple{3,Vector{BlockRBMatAlgebraicContribution{T}}}},args...) where T
  S = BlockRBMatAlgebraicContribution{T}
  njacs = num_active_dirs(rbinfo.rb_path)
  ad_jacs_lin = Vector{S}(undef,njacs)
  ad_jacs_nlin = Vector{S}(undef,njacs)
  ad_jacs_aux = Vector{S}(undef,njacs)
  for i = 1:njacs
    path_lin = joinpath(rbinfo.rb_path,"rb_lhs_lin_$i")
    path_nlin = joinpath(rbinfo.rb_path,"rb_lhs_nlin_$i")
    path_aux = joinpath(rbinfo.rb_path,"rb_lhs_aux_$i")
    ad_jacs_lin[i] = load_algebraic_contrib(path_lin,S,args...)
    ad_jacs_nlin[i] = load_algebraic_contrib(path_nlin,S,args...)
    ad_jacs_aux[i] = load_algebraic_contrib(path_aux,S,args...)
  end
  ad_jacs_lin,ad_jacs_nlin,ad_jacs_aux
end

function collect_compress_rhs(
  rbinfo::BlockRBInfo,
  op::PTOperator,
  rbspace::BlockRBSpace{T}) where T

  nblocks = get_nblocks(rbspace)
  blocks = Vector{RBVecAlgebraicContribution{T}}(undef,nblocks)
  touched = Vector{Bool}(undef,nblocks)
  for row = 1:nblocks
    op_row_col = op[row,:]
    touched[row] = check_touched_residuals(op_row_col)
    if touched[row]
      rbinfo_row = rbinfo[row]
      rbspace_row = rbspace[row]
      blocks[row] = collect_compress_rhs(rbinfo_row,op_row_col,rbspace_row)
    end
  end

  BlockRBVecAlgebraicContribution(blocks,touched)
end

function collect_compress_lhs(
  rbinfo::BlockRBInfo,
  op::PTOperator,
  rbspace::BlockRBSpace{T};
  θ::Real=1) where T

  nblocks = get_nblocks(rbspace)
  njacs = length(op.odeop.feop.jacs)
  ad_jacs = Vector{BlockRBMatAlgebraicContribution{T}}(undef,njacs)
  for i = 1:njacs
    touched_i = Matrix{Bool}(undef,nblocks,nblocks)
    blocks_i = Matrix{RBMatAlgebraicContribution{T}}(undef,nblocks,nblocks)
    for (row,col) = index_pairs(nblocks,nblocks)
      op_row_col = op[row,col]
      touched_i[row,col] = check_touched_jacobians(op_row_col;i)
      if touched_i[row,col]
        rbinfo_col = rbinfo[col]
        rbspace_row = rbspace[row]
        rbspace_col = rbspace[col]
        blocks_i[row,col] = _collect_compress_lhs(rbinfo_col,op_row_col,rbspace_row,rbspace_col;i,θ)
      end
    end
    ad_jacs[i] = BlockRBMatAlgebraicContribution(blocks_i,touched_i)
  end

  return ad_jacs
end

function check_touched_residuals(op::PTOperator)
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
  int = integrate(res(μ1,t1,xh1,dv))
  return !isnothing(int)
end

function check_touched_jacobians(op::PTOperator;i=1)
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
  rbinfo::BlockRBInfo,
  op::PTOperator,
  rbres::BlockRBVecAlgebraicContribution{T},
  rbspace::BlockRBSpace{T}) where T

  nblocks = get_nblocks(rbres)
  blocks = Vector{PTArray{Vector{T}}}(undef,nblocks)
  for row = 1:nblocks
    cache_row = cache_at_index(cache,rbspace,row)
    op_row_col = op[row,:]
    rbspace_row = rbspace[row]
    if rbres.touched[row]
      rbinfo_row = rbinfo[row]
      blocks[row] = collect_rhs_contributions!(
        cache_row,rbinfo_row,op_row_col,rbres[row],rbspace_row)
    else
      nrow = get_rb_ndofs(rbspace_row)
      blocks[row] = AffinePTArray(zeros(T,nrow),length(op.μ))
    end
  end
  vcat(blocks...)
end

function collect_lhs_contributions!(
  cache,
  rbinfo::BlockRBInfo,
  op::PTOperator,
  rbjacs::Vector{BlockRBMatAlgebraicContribution{T}},
  rbspace::BlockRBSpace{T}) where T

  njacs = length(rbjacs)
  nblocks = get_nblocks(testitem(rbjacs))
  rb_jacs_contribs = Vector{PTArray{Matrix{T}}}(undef,njacs)
  for i = 1:njacs
    rb_jac_i = rbjacs[i]
    blocks = Matrix{PTArray{Matrix{T}}}(undef,nblocks,nblocks)
    for (row,col) = index_pairs(nblocks,nblocks)
      cache_row_col = cache_at_index(cache,rbspace,row,col)
      op_row_col = op[row,col]
      rbspace_row = rbspace[row]
      rbspace_col = rbspace[col]
      if rb_jac_i.touched[row,col]
        rbinfo_col = rbinfo[col]
        blocks[row,col] = collect_lhs_contributions!(
          cache_row_col,rbinfo_col,op_row_col,rb_jac_i[row,col],rbspace_row,rbspace_col;i)
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

function cache_at_index(cache,rbspace::RBBlock,args...)
  coeff_cache,rb_cache = cache
  (a,b),solve_cache = coeff_cache
  offsets = fe_offsets(rbspace)
  aidx = get_at_offsets(a,offsets,args...)
  return ((aidx,b),solve_cache),rb_cache
end
