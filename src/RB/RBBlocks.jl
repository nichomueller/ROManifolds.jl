abstract type RBBlock{T,N} end

Base.getindex(b::RBBlock,i...) = b.blocks[i...]
Base.iterate(b::RBBlock,args...) = iterate(b.blocks,args...)
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

const AbstractSnapshots{T} = Union{Snapshots{T},BlockSnapshots{T}}

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

const AbstractRBSpace{T} = Union{RBSpace{T},BlockRBSpace{T}}

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
  blocks = map(index_pairs(1,nblocks)) do (row,col)
    feop_row_col = feop[row,col]
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
  BlockRBSpace(bases_space,bases_time)
end

function add_space_supremizers(
  bases_space::Vector{<:Matrix},
  feop::PTFEOperator,
  norm_matrix::AbstractVector,
  args...)

  bs_primal,bs_dual... = bases_space
  nm_primal, = norm_matrix
  dual_nfields = length(bs_dual)
  for (row,col) in index_pairs(1,dual_nfields)
    println("Computing supremizers in space for dual field $col")
    feop_row_col = feop[row,col+1]
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
  touched::Array{Bool,N}

  function BlockRBAlgebraicContribution(
    blocks::Array{RBAlgebraicContribution{T},N},
    touched::Array{Bool,N}) where {T,N}

    new{T,N}(blocks,touched)
  end
end

const AbstractRBAlgebraicContribution{T,N} = Union{RBAlgebraicContribution{T},BlockRBAlgebraicContribution{T,N}}

function collect_compress_rhs_lhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbspace::BlockRBSpace,
  snaps::BlockSnapshots,
  μ::Table)

  nblocks = get_nblocks(rbspace)
  nsnaps = info.nsnaps_system
  snapsθ = recenter(fesolver,snaps,μ)
  _μ = μ[1:nsnaps]
  _snapsθ = map(1:nblocks) do row
    snapsθ_row = snapsθ[row]
    snapsθ_row[1:nsnaps]
  end
  rhs = collect_compress_rhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  lhs = collect_compress_lhs(info,feop,fesolver,rbspace,_snapsθ,_μ)
  rhs,lhs
end

function collect_compress_rhs(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbspace::BlockRBSpace,
  snaps::Vector{<:PTArray},
  μ::Table)

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
      rbres = testvalue(RBAlgebraicContribution,feop;vector=true)
    end
    rbres,touched
  end
  blocks = first.(result)
  touched = last.(result)
  return BlockRBAlgebraicContribution(blocks,touched)
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
        rbjac = testvalue(RBAlgebraicContribution,feop;vector=false)
      end
      rbjac,touched
    end
    blocks_i = first.(result_i)
    touched_i = last.(touched_i)
    ad_jacs[i] = BlockRBAlgebraicContribution(blocks_i,touched_i)
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
  dv = get_fe_basis(test)
  int = feop.res(μ1,t1,uh1,dv)
  return isnothing(int)
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
  dv = get_fe_basis(test)
  du = get_trial_fe_basis(trial(nothing,nothing))
  int = feop.jacs[i](μ1,t1,uh1,dv,du)
  return isnothing(int)
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
    rbres_row = rbres[row]
    if rbres_row.touched
      collect_rhs_contributions!(cache,info,feop_row,fesolver,rbres_row,vsnaps,args...)
    else
      rbspace_row = rbspace[row]
      allocate_vector(rbspace_row)
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
      rb_jac_i_row_col = rb_jac_i[row,col]
      if rb_jac_i_row_col.touched
        collect_lhs_contributions!(cache,info,feop_row_col,fesolver,rb_jac_i_row_col,sols_col,args...;i)
      else
        rbspace_row = rbspace[row]
        rbspace_col = rbspace[col]
        allocate_matrix(rbspace_row,rbspace_col)
      end
    end
    rb_jacs_contribs[i] = hvcat(blocks...)
  end
  return sum(rb_jacs_contribs)
end
