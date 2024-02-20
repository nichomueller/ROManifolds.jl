function reduced_fe_space(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::AbstractTransientSnapshots)

  trial = get_trial(feop)
  test = get_test(feop)
  norm_matrix = get_norm_matrix(info,feop)
  soff = select_snapshots(s,offline_params(info))
  basis_space,basis_time = reduced_basis(soff,norm_matrix;ϵ=get_tol(info))
  reduced_trial = RBSpace(trial,basis_space,basis_time)
  reduced_test = RBSpace(test,basis_space,basis_time)
  return reduced_trial,reduced_test
end

function reduced_basis(s::AbstractTransientSnapshots,args...;kwargs...)
  basis_space,basis_time = compute_bases(s,args...;kwargs...)
  return basis_space,basis_time
end

function reduced_basis(s::TransientSnapshotsWithDirichletValues,args...;kwargs...)
  basis_space,basis_time = compute_bases(soff_free,args...;kwargs...)
  ranks = size(basis_space,2),size(basis_time,2)
  sdir = get_dirichlet_values(s)
  basis_space_dir,basis_time_dir = compute_bases(sdir,args...;ranks)
  return basis_space,basis_time
end

function reduced_basis(s::NnzSnapshots,args...;kwargs...)
  basis_space,basis_time = compute_bases(s,args...;kwargs...)
  sparse_basis_space = recast(basis_space,s)
  return sparse_basis_space,basis_time
end

function compute_bases(
  s::AbstractTransientSnapshots,
  norm_matrix=nothing;
  kwargs...)

  flag,s = _return_flag(s)
  b1 = tpod(s,norm_matrix;kwargs...)
  compressed_s = compress(b1,s)
  compressed_s = change_mode(compressed_s)
  b2 = tpod(compressed_s;kwargs...)
  _return_bases(flag,b1,b2)
end

function _return_flag(s)
  flag = false
  if size(s,1) < size(s,2)
    s = change_mode(s)
    flag = true
  end
  flag,s
end

function _return_bases(flag,b1,b2)
  if flag
    basis_space,basis_time = b2,b1
  else
    basis_space,basis_time = b1,b2
  end
  basis_space,basis_time
end

struct RBSpace{S,BS,BT} <: FESpace
  space::S
  basis_space::BS
  basis_time::BT
end

function Arrays.evaluate(U::RBSpace,args...)
  space = evaluate(U.space,args...)
  RBSpace(space,U.basis_space,U.basis_time)
end

(U::RBSpace)(r) = evaluate(U,r)
(U::RBSpace)(μ,t) = evaluate(U,μ,t)

ODETools.∂t(U::RBSpace) = RBSpace(∂t(U),U.basis_space,U.basis_time)
ODETools.∂tt(U::RBSpace) = RBSpace(∂tt(U),U.basis_space,U.basis_time)

function get_basis_space end
get_basis_space(r::RBSpace) = r.basis_space
num_space_dofs(r::RBSpace) = size(get_basis_space(r),1)
function num_reduced_space_dofs end
num_reduced_space_dofs(r::RBSpace) = size(get_basis_space(r),2)

function get_basis_time end
get_basis_time(r::RBSpace) = r.basis_time
FEM.num_times(r::RBSpace) = size(get_basis_time(r),1)
function num_reduced_times end
num_reduced_times(r::RBSpace) = size(get_basis_time(r),2)

FESpaces.num_free_dofs(r::RBSpace) = num_reduced_space_dofs(r)*num_reduced_times(r)

FESpaces.get_free_dof_ids(r::RBSpace) = Base.OneTo(num_free_dofs(r))

FESpaces.get_dirichlet_dof_ids(r::RBSpace) = get_dirichlet_dof_ids(r.space)

FESpaces.num_dirichlet_dofs(r::RBSpace) = num_dirichlet_dofs(r.space)

FESpaces.num_dirichlet_tags(r::RBSpace) = num_dirichlet_tags(r.space)

FESpaces.get_dirichlet_dof_tag(r::RBSpace) = get_dirichlet_dof_tag(r.space)

function FESpaces.get_vector_type(r::RBSpace)
  change_length(x) = x
  change_length(::Type{ParamVector{T,A,L}}) where {T,A,L} = ParamVector{T,A,Int(L/num_times(r))}
  V = get_vector_type(r.space)
  newV = change_length(V)
  return newV
end

function compress(r::RBSpace,xmat::AbstractMatrix)
  basis_space = get_basis_space(r)
  basis_time = get_basis_time(r)

  red_xmat = (basis_space'*xmat)*basis_time
  x = vec(red_xmat')
  return x
end

function compress(
  trial::RBSpace,
  test::RBSpace,
  xmat::AbstractMatrix{T};
  combine=(x,y)->x) where T

  basis_space_test = get_basis_space(test)
  basis_time_test = get_basis_time(test)
  basis_space_trial = get_basis_space(trial)
  basis_time_trial = get_basis_time(trial)
  ns_test,ns_trial = size(basis_space_test,2),size(basis_space_trial,2)
  nt_test,nt_trial = size(basis_time_test,2),size(basis_time_trial,2)

  red_xvec = compress_basis_space(xmat,trial,test)
  red_xmat = stack(vec.(red_xvec))'  # Nt x ns_test*ns_trial
  st_proj = zeros(T,nt_test,nt_trial,ns_test*ns_trial)
  st_proj_shift = zeros(T,nt_test,nt_trial,ns_test*ns_trial)
  @inbounds for ins = 1:ns_test*ns_trial, jt = 1:nt_trial, it = 1:nt_test
    st_proj[it,jt,ins] = sum(basis_time_test[:,it].*basis_time_trial[:,jt].*red_xmat[:,ins])
    st_proj_shift[it,jt,ins] = sum(basis_time_test[2:end,it].*basis_time_trial[1:end-1,jt].*red_xmat[2:end,ins])
  end
  st_proj = combine(st_proj,st_proj_shift)
  st_proj_mat = zeros(T,ns_test*nt_test,ns_trial*nt_trial)
  @inbounds for i = 1:ns_trial, j = 1:ns_test
    st_proj_mat[1+(j-1)*nt_test:j*nt_test,1+(i-1)*nt_trial:i*nt_trial] = st_proj[:,:,(i-1)*ns_test+j]
  end
  return st_proj_mat
end

function recast(r::RBSpace,red_x::AbstractVector)
  basis_space = get_basis_space(r)
  basis_time = get_basis_time(r)
  ns = num_reduced_space_dofs(r)
  nt = num_reduced_times(r)

  red_xmat = reshape(red_x,nt,ns)
  xmat = basis_space*(basis_time*red_xmat)'
  x = eachcol(xmat) |> collect
  ParamArray(x)
end

function recast(r::RBSpace,red_x::ParamVector)
  map(red_x) do red_x
    recast(r,red_x)
  end
end

# multi field interface
const BlockRBSpace = RBSpace{S,BS,BT} where {
  S,BS<:AbstractVector{<:AbstractMatrix},BT<:AbstractVector{<:AbstractMatrix}}

function reduced_fe_space(
  info::BlockRBInfo,
  feop::TransientParamFEOperator,
  s::BlockSnapshots)

  trial = get_trial(feop)
  test = get_test(feop)
  norm_matrix = get_norm_matrix(info,feop)
  soff = select_snapshots(s,offline_params(info))
  basis_space,basis_time = map(blocks(soff),norm_matrix) do soff,norm_matrix
    reduced_basis(soff,norm_matrix;ϵ=get_tol(info))
  end |> tuple_of_arrays
  if info.compute_supremizers
    supr_op = compute_supremizer_operator(feop)
    basis_space = add_space_supremizers(basis_space,supr_op,norm_matrix)
    basis_time = add_time_supremizers(basis_time)
  end
  reduced_trial = RBSpace(trial,basis_space,basis_time)
  reduced_test = RBSpace(test,basis_space,basis_time)
  return reduced_trial,reduced_test
end

function add_space_supremizers(basis_space,supr_op,norm_matrix)
  @check length(basis_space) == 2 "Have to extend this if dealing with more than 2 equations"
  basis_space_primal,basis_space_dual = basis_space
  norm_matrix_primal = first(norm_matrix)
  supr_i = supr_op * basis_space_dual
  gram_schmidt!(supr_i,basis_space_primal,norm_matrix_primal)
  basis_space_primal = hcat(basis_space_primal,supr_i)
  return basis_space_primal,basis_space_dual
end

function add_time_supremizers(basis_time;kwargs...)
  basis_time_primal,basis_time_dual... = basis_time
  for bdual in basis_time_dual
    basis_time_primal = add_time_supremizers(basis_time_primal,bdual;kwargs...)
  end
  return basis_time_primal,basis_time_dual...
end

function add_time_supremizers(basis_primal,basis_dual;tol=1e-2)
  basis_pd = basis_primal'*basis_dual

  function enrich(basis_primal,basis_pd,v)
    vnew = copy(v)
    orth_complement!(vnew,basis_primal)
    vnew /= norm(vnew)
    hcat(basis_primal,vnew),vcat(basis_pd,vnew'*basis_dual)
  end

  count = 0
  ntp_minus_ntu = size(basis_dual,2) - size(basis_primal,2)
  if ntp_minus_ntu > 0
    for ntp = 1:ntp_minus_ntu
      basis_primal,basis_pd = enrich(basis_primal,basis_pd,basis_dual[:,ntp])
      count += 1
    end
  end

  ntp = 1
  while ntp ≤ size(basis_pd,2)
    proj = ntp == 1 ? zeros(size(basis_pd[:,1])) : orth_projection(basis_pd[:,ntp],basis_pd[:,1:ntp-1])
    dist = norm(basis_pd[:,1]-proj)
    if dist ≤ tol
      basis_primal,basis_pd = enrich(basis_primal,basis_pd,basis_dual[:,ntp])
      count += 1
      ntp = 0
    else
      basis_pd[:,ntp] .-= proj
    end
    ntp += 1
  end

  basis_primal
end

function BlockArrays.blocks(r::BlockRBSpace)
  map(r.space.spaces,r.basis_space,r.basis_time) do fs,bs,bt
    RBSpace(fs,bs,bt)
  end
end

function recast(r::BlockRBSpace,red_x::ParamBlockVector)
  block_red_x = map(blocks(r),blocks(red_x)) do r,red_x
    recast(r,red_x)
  end
  mortar(block_red_x)
end
