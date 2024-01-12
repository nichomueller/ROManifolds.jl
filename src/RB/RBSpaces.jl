struct RBSpace{T}
  basis_space::Matrix{T}
  basis_time::Matrix{T}
end

get_basis_space(rb::RBSpace) = rb.basis_space
get_basis_time(rb::RBSpace) = rb.basis_time
num_space_dofs(rb::RBSpace) = size(rb.basis_space,1)
FEM.num_time_dofs(rb::RBSpace) = size(rb.basis_time,1)
num_rb_space_ndofs(rb::RBSpace) = size(rb.basis_space,2)
num_rb_time_ndofs(rb::RBSpace) = size(rb.basis_time,2)
num_rb_ndofs(rb::RBSpace) = num_rb_time_ndofs(rb)*num_rb_space_ndofs(rb)
FESpaces.get_vector_type(::RBSpace{T}) where T = Vector{T}

function Utils.save(rbinfo::RBInfo,rb::RBSpace)
  path = joinpath(rbinfo.rb_path,"rb")
  save(path,rb)
end

function Utils.load(rbinfo::RBInfo,T::Type{RBSpace{S}}) where S
  path = joinpath(rbinfo.rb_path,"rb")
  load(path,T)
end

function reduced_basis(rbinfo::RBInfo,feop::PTFEOperator,snaps::Snapshots)
  println("Computing RB space")
  ϵ = rbinfo.ϵ
  nsnaps_state = rbinfo.nsnaps_state
  norm_matrix = get_norm_matrix(rbinfo,feop)
  return reduced_basis(snaps,norm_matrix;ϵ,nsnaps_state)
end

function reduced_basis(
  snaps::Snapshots,
  norm_matrix;
  nsnaps_state=50,
  kwargs...)

  nzm = NnzMatrix(snaps[1:nsnaps_state];nparams=nsnaps_state)
  basis_space_nnz,basis_time = compress(nzm,norm_matrix;kwargs...)
  basis_space = recast(basis_space_nnz)
  return RBSpace(basis_space,basis_time)
end

function recast(x::AbstractVector,rb::RBSpace)
  basis_space = get_basis_space(rb)
  basis_time = get_basis_time(rb)
  ns_rb = num_rb_space_ndofs(rb)
  nt_rb = num_rb_time_ndofs(rb)

  x_mat = reshape(x,nt_rb,ns_rb)
  xrb_mat = basis_space*(basis_time*x_mat)'
  xrb = [xrb_mat[:,i] for i = axes(xrb_mat,2)]
  return xrb
end

function recast(x::PTArray,rb::RBSpace)
  array = map(eachindex(x)) do xi
    recast(xi,rb)
  end
  PTArray(array...)
end

function space_time_projection(x::PTArray,rb::RBSpace)
  time_ndofs = num_time_dofs(rb)
  nparams = Int(length(x)/time_ndofs)
  array = map(1:nparams) do np
    x_np = stack(x[(np-1)*time_ndofs+1:np*time_ndofs])
    space_time_projection(x_np,rb)
  end
  PTArray(array)
end

function space_time_projection(mat::AbstractMatrix,rb::RBSpace)
  basis_space = get_basis_space(rb)
  basis_time = get_basis_time(rb)
  st_proj = (basis_space'*mat)*basis_time
  return vec(st_proj')
end

function space_time_projection(nzm::NnzMatrix,rb::RBSpace)
  mat = recast(nzm)
  space_time_projection(mat,rb)
end

function space_time_projection(
  nzm::NnzMatrix{T},rb_row::RBSpace,rb_col::RBSpace;combine_projections=(x,y)->x) where T
  basis_space_row = get_basis_space(rb_row)
  basis_time_row = get_basis_time(rb_row)
  basis_space_col = get_basis_space(rb_col)
  basis_time_col = get_basis_time(rb_col)
  ns_row,ns_col = size(basis_space_row,2),size(basis_space_col,2)
  nt_row,nt_col = size(basis_time_row,2),size(basis_time_col,2)

  s_proj = compress(basis_space_row,basis_space_col,nzm)
  s_proj_mat = stack([vec(x) for x in s_proj])'  # time_ndofs x ns_row*ns_col
  st_proj_center = zeros(T,nt_row,nt_col,ns_row*ns_col)
  st_proj_shift = zeros(T,nt_row,nt_col,ns_row*ns_col)
  @inbounds for ins = 1:ns_row*ns_col, jt = 1:nt_col, it = 1:nt_row
    st_proj_center[it,jt,ins] = sum(basis_time_row[:,it].*basis_time_col[:,jt].*s_proj_mat[:,ins])
    st_proj_shift[it,jt,ins] = sum(basis_time_row[2:end,it].*basis_time_col[1:end-1,jt].*s_proj_mat[2:end,ins])
  end
  st_proj = combine_projections(st_proj_center,st_proj_shift)
  st_proj_mat = zeros(T,ns_row*nt_row,ns_col*nt_col)
  @inbounds for i = 1:ns_col, j = 1:ns_row
    st_proj_mat[1+(j-1)*nt_row:j*nt_row,1+(i-1)*nt_col:i*nt_col] = st_proj[:,:,(i-1)*ns_row+j]
  end
  return st_proj_mat
end

function project_recast(snap::PTArray,rb::RBSpace)
  mat = stack(snap.array)
  rb_proj = space_time_projection(mat,rb)
  array = recast(rb_proj,rb)
  PTArray(array)
end

function TransientFETools.get_algebraic_operator(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  rbspace::RBSpace{T},
  params::Table) where T

  dtθ = fesolver.θ == 0.0 ? fesolver.dt : fesolver.dt*fesolver.θ
  times = get_stencil_times(fesolver)
  bs = get_basis_space(rbspace)
  ns = num_rb_space_ndofs(rbspace)

  ode_cache = allocate_cache(feop,params,times)
  ode_cache = update_cache!(ode_cache,feop,params,times)
  N = length(times)*length(params)
  array = Vector{Vector{T}}(undef,N)
  @inbounds for n = 1:N
    col = mod(n,ns) == 0 ? ns : mod(n,ns)
    array[n] = bs[:,col]
  end
  sols = PTArray(array)
  sols_cache = zero(sols)
  get_algebraic_operator(feop,params,times,dtθ,sols,ode_cache,sols_cache)
end
