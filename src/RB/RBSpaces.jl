abstract type AbstractRBSpace{T} end

get_basis_space(rb::AbstractRBSpace) = rb.basis_space
get_basis_time(rb::AbstractRBSpace) = rb.basis_time

function Algebra.allocate_vector(rb::AbstractRBSpace{T}) where T
  zeros(T,num_rb_dofs(rb))
end

function Algebra.allocate_vector(rb::AbstractRBSpace{T}) where T
  zeros(T,num_rb_dofs(rb))
end

function save(info::RBInfo,rb::AbstractRBSpace)
  if info.save_structures
    path = joinpath(info.rb_path,"rb")
    save(path,rb)
  end
end

function load(info::RBInfo,T::Type{AbstractRBSpace})
  path = joinpath(info.rb_path,"rb")
  load(path,T)
end

struct RBSpace{T} <: AbstractRBSpace{T}
  basis_space::Matrix{T}
  basis_time::Matrix{T}

  function RBSpace(
    basis_space::Matrix{T},
    basis_time::Matrix{T}) where T
    new{T}(basis_space,basis_time)
  end
end

function num_rb_dofs(rb::RBSpace)
  size(rb.basis_space,2)*size(rb.basis_time,2)
end

function reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::Snapshots,
  args...)

  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix(feop,energy_norm)
  basis_space_nnz,basis_time = compress(info,feop,snaps,norm_matrix,args...)
  basis_space = recast(basis_space_nnz)
  RBSpace(basis_space,basis_time)
end

function recast(rb::RBSpace,x::AbstractVector)
  basis_space = get_basis_space(rb)
  basis_time = get_basis_time(rb)
  ns_rb = size(basis_space,2)
  nt_rb = size(basis_time,2)

  x_mat_i = reshape(x,nt_rb,ns_rb)
  return basis_space*(basis_time*x_mat_i)'
end

function recast(rb::RBSpace,x::PTArray{T}) where T
  basis_space = get_basis_space(rb)
  basis_time = get_basis_time(rb)
  ns_rb = size(basis_space,2)
  nt_rb = size(basis_time,2)

  n = length(x)
  array = Vector{Matrix{eltype(T)}}(undef,n)
  @inbounds for i = 1:n
    x_mat_i = reshape(x[i],nt_rb,ns_rb)
    array[i] = basis_space*(basis_time*x_mat_i)'
  end

  PTArray(array)
end

for f in (:space_time_projection,:test_reduced_basis)
  @eval begin
    function $f(s::Snapshots,rb::RBSpace...;n=1)
      snap = s[n]
      $f(snap,rb...)
    end

    function $f(snap::PTArray{<:AbstractVector},rb::RBSpace...)
      mat = hcat(get_array(snap)...)
      $f(mat,rb...)
    end
  end
end

function space_time_projection(mat::AbstractMatrix,rb::RBSpace)
  basis_space = get_basis_space(rb)
  basis_time = get_basis_time(rb)
  st_proj = (basis_space'*mat)*basis_time
  return reshape(st_proj',:)
end

function space_time_projection(
  mat::NnzMatrix{T},rb_row::RBSpace,rb_col::RBSpace,combine_projections=(x,y)->x) where T
  basis_space_row = get_basis_space(rb_row)
  basis_time_row = get_basis_time(rb_row)
  basis_space_col = get_basis_space(rb_col)
  basis_time_col = get_basis_time(rb_col)
  ns_row,ns_col = size(basis_space_row,2),size(basis_space_col,2)
  nt_row,nt_col = size(basis_time_row,2),size(basis_time_col,2)

  s_proj = compress(basis_space_row,basis_space_col,mat)
  s_proj_mat = hcat(s_proj...)' # time_ndofs x ns_row*ns_col
  st_proj_center = zeros(T,nt_row,nt_col,ns_row*ns_col)
  st_proj_shift = zeros(T,nt_row,nt_col,ns_row*ns_col)
  @inbounds for ins = 1:ns_row*ns_col, jt = 1:nt_col, it = 1:nt_row
    st_proj_center[it,jt,ins] = sum(basis_time_row[:,it].*basis_time_col[:,jt].*s_proj_mat[:,ins])
    st_proj_shift[it,jt,ins] = sum(basis_time_row[2:end,it].*basis_time_col[1:end-1,jt].*s_proj_mat[2:end,ins])
  end
  st_proj = combine_projections(st_proj_center,st_proj_shift)
  st_proj_mat = zeros(T,ns_row*nt_row,ns_col*nt_col)
  @inbounds for i = 1:ns_row, j = 1:ns_col
    st_proj_mat[1+(j-1)*nt_row:j*nt_row,1+(i-1)*nt_col:i*nt_col] = st_proj[:,:,(i-1)*ns_row+j]
  end
  return st_proj_mat
end

function test_reduced_basis(mat::AbstractMatrix,rb::RBSpace)
  rb_proj = space_time_projection(mat,rb)
  rb_approx = recast(rb,rb_proj)
  err = maximum(abs.(mat-rb_approx))
  println("RB approximation error in infty norm of snapshot $n = $err")
  return rb_proj
end

# Multifield interface
struct BlockRBSpace{T} <: AbstractRBSpace{T}
  basis_space::Vector{Matrix{T}}
  basis_time::Vector{Matrix{T}}

  function BlockRBSpace(
    basis_space::Vector{Matrix{T}},
    basis_time::Vector{Matrix{T}}) where T
    new{T}(basis_space,basis_time)
  end
end

get_nfields(rb::BlockRBSpace) = length(rb.basis_space)
Base.getindex(rb::BlockRBSpace,i...) = RBSpace(rb.basis_space[i...],rb.basis_time[i...])

function num_rb_dofs(rb::BlockRBSpace)
  nfields = get_nfields(rb)
  ndofs = 0
  @inbounds for i = 1:nfields
    ndofs += num_rb_dofs(rb[i])
  end
  ndofs
end

function reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  args...)

  energy_norm = info.energy_norm
  norm_matrix = get_norm_matrix.(energy_norm,feop)
  basis_space_nnz,basis_time = compress(info,feop,snaps,norm_matrix,args...)
  basis_space = recast(basis_space_nnz)
  BlockRBSpace(basis_space,basis_time)
end

function compress(info::RBInfo,::PTFEOperator,snaps,args...)
  nzm = NnzArray(snaps)
  ϵ = info.ϵ
  steady = num_time_dofs(nzm) == 1 ? SteadyPOD() : DefaultPOD()
  compress(nzm,steady,args...;ϵ)
end

function compress(
  info::RBInfo,
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  args...;
  compute_supremizers=false,
  kwargs...)

  nzm = NnzArray(snaps)
  nfields = get_nfields(nzm)
  all_idx = index_pairs(nfields,1)
  rb = map(all_idx) do i
    feopi = filter_operator(feop,i)
    compress(info,feopi,nzm[i])
  end
  bases_space = map(get_basis_space,rb)
  bases_time = map(get_basis_time,rb)
  if compute_supremizers
    bases_space = add_space_supremizers(bases_space,feop,snaps,args...)
    bases_time = add_time_supremizers(bases_time;kwargs...)
  end
  BlockRBSpace(bases_space,bases_time)
end

function add_space_supremizers(
  bases_space::Vector{<:Matrix},
  feop::PTFEOperator,
  snaps::BlockSnapshots,
  norm_matrix,
  args...)

  bs_primal,bs_dual... = bases_space
  n_dual_fields = length(bs_dual)
  all_idx = index_pairs(n_dual_fields,1)
  for idx in all_idx
    println("Computing supremizers in space for dual field $idx")
    feop_i = filter_operator(feop,idx)
    supr_i = space_supremizers(bs_dual[idx],feop_i,snaps[idx],args...)
    orth_supr_i = gram_schmidt(supr_i,bs_primal,norm_matrix)
    bs_primal = hcat(bs_primal,orth_supr_i)
  end
  return bs_primal,bs_dual
end

function space_supremizers(
  basis_space::Matrix,
  feop::PTFEOperator,
  snaps::Snapshots,
  fesolver::PODESolver,
  args...)

  constraint_mat = collect_jacobians(fesolver,feop,snaps,args...)
  if length(constraint_mat) == 1
    return constraint_mat*basis_space
  else
    @assert length(constraint_mat) == length(snaps)
    return map(*,constraint_mat,snaps)
  end
end

function add_time_supremizers(bases_time::Vector{<:Matrix};ttol::Real)
  bt_primal,bt_dual... = bases_time
  n_dual_fields = length(bt_dual)
  all_idx = index_pairs(n_dual_fields,1)
  for idx in all_idx
    println("Computing supremizers in time for dual field $idx")
    supr_i = add_time_supremizers(bt_primal,bt_dual[idx];ttol)
    append!(bt_primal,supr_i)
  end
  return bt_primal,btdual
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

function filter_rbspace(rb::BlockRBSpace,idx::Int)
  basis_time = get_basis_space(rb)[idx]
  basis_time = get_basis_time(rb)[idx]
  RBSpace(basis_space,basis_time)
end
