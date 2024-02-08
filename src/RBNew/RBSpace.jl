function reduced_fe_space(
  rbinfo::RBInfo,
  feop::TransientParamFEOperator,
  s::AbstractTransientSnapshots)

  trial = get_trial(feop)
  test = get_test(feop)
  ϵ = get_tol(rbinfo)
  nsnaps_offline = num_offline_params(rbinfo)
  norm_matrix = get_norm_matrix(rbinfo,feop)
  soff = select_snapshots(s,1:nsnaps_offline)
  basis_space,basis_time = reduced_basis(soff,norm_matrix;ϵ)
  reduced_trial = TrialRBSpace(trial,basis_space,basis_time)
  reduced_test = TestRBSpace(test,basis_space,basis_time)
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

abstract type RBSpace{S} <: FESpace end

struct TestRBSpace{S,B} <: RBSpace{S}
  space::S
  basis_space::B
  basis_time::B
end

struct TrialRBSpace{S,B} <: RBSpace{S}
  space::S
  basis_space::B
  basis_time::B
end

(test::TestRBSpace)(r) = test
(test::TestRBSpace)(μ,t) = test
(trial::TrialRBSpace)(r) = TrialRBSpace(trial.space(r),trial.basis_space,trial.basis_time)
(trial::TrialRBSpace)(μ,t) = TrialRBSpace(trial.space(μ,t),trial.basis_space,trial.basis_time)

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

function get_free_reduced_dof_ids end
get_free_reduced_dof_ids(r::RBSpace) = Base.OneTo(num_reduced_space_dofs(r))

function num_free_reduced_dofs end
num_free_reduced_dofs(r::RBSpace) = length(get_free_dof_ids(r))

function zero_free_reduced_values end
function zero_free_reduced_values(r::RBSpace)
  V = get_vector_type(r)
  allocate_vector(V,num_free_dofs(r))
end

FESpaces.get_free_dof_ids(r::RBSpace) = get_free_dof_ids(r.space)

FESpaces.get_triangulation(r::RBSpace) = get_triangulation(r.space)

FESpaces.get_dof_value_type(r::RBSpace) = get_dof_value_type(r.space)

FESpaces.get_vector_type(r::RBSpace) = get_vector_type(r.space)

FESpaces.get_cell_dof_ids(r::RBSpace) = get_cell_dof_ids(r.space)

FESpaces.get_fe_basis(r::RBSpace) = get_fe_basis(r.space)

FESpaces.get_trial_fe_basis(r::RBSpace) = get_trial_fe_basis(r.space)

FESpaces.get_fe_dof_basis(r::RBSpace) = get_fe_dof_basis(r.space)

FESpaces.ConstraintStyle(::Type{<:RBSpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_cell_isconstrained(r::RBSpace) = get_cell_isconstrained(r.space)

FESpaces.get_cell_constraints(r::RBSpace) = get_cell_constraints(r.space)

FESpaces.get_dirichlet_dof_ids(r::RBSpace) = get_dirichlet_dof_ids(r.space)

FESpaces.get_cell_is_dirichlet(r::RBSpace) = get_cell_is_dirichlet(r.space)

FESpaces.num_dirichlet_dofs(r::RBSpace) = num_dirichlet_dofs(r.space)

FESpaces.num_dirichlet_tags(r::RBSpace) = num_dirichlet_tags(r.space)

FESpaces.get_dirichlet_dof_tag(r::RBSpace) = get_dirichlet_dof_tag(r.space)

FESpaces.scatter_free_and_dirichlet_values(r::RBSpace,fv,dv) = scatter_free_and_dirichlet_values(r.space,fv,dv)

FESpaces.gather_free_and_dirichlet_values(r::RBSpace,cv) = gather_free_and_dirichlet_values(r.space,cv)

FESpaces.gather_free_and_dirichlet_values!(fv,dv,r::RBSpace,cv) = gather_free_and_dirichlet_values!(fv,dv,r.space,cv)

FESpaces.gather_dirichlet_values(r::RBSpace,cv) = gather_dirichlet_values(r.space,cv)

FESpaces.gather_dirichlet_values!(dv,r::RBSpace,cv) = gather_dirichlet_values!(dv,r.space,cv)

FESpaces.gather_free_values(r::RBSpace,cv) = gather_free_values(r.space,cv)

FESpaces.gather_free_values!(fv,r::RBSpace,cv) = gather_free_values!(fv,r.space,cv)

# function FESpaces.FEFunction(
#   fs::RBSpace,free_values::ParamArray,dirichlet_values::ParamArray)
#   cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
#   cell_field = CellField(fs,cell_vals)
#   SingleFieldParamFEFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)
# end

function compress(r::RBSpace,xmat::AbstractMatrix)
  basis_space = get_basis_space(r)
  basis_time = get_basis_time(r)

  red_xmat = (basis_space'*xmat)*basis_time
  x = vec(red_xmat')
  return x
end

function compress(
  trial::TrialRBSpace,
  test::TestRBSpace,
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

function recast(r::RBSpace,x::AbstractVector)
  basis_space = get_basis_space(r)
  basis_time = get_basis_time(r)
  ns = num_reduced_space_dofs(rb)
  nt = num_reduced_times(rb)

  xmat = reshape(x,nt,ns)
  return basis_space*(basis_time*xmat)'
end
