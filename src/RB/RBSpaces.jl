struct RBSpace{T}
  basis_space::Matrix{T}
  basis_time::Matrix{T}

  function RBSpace(
    basis_space::Matrix{T},
    basis_time::Matrix{T}) where T
    new{T}(basis_space,basis_time)
  end
end

function Base.show(io::IO,rb::RBSpace)
  nbs = size(rb.basis_space,2)
  nbt = size(rb.basis_time,2)
  printstyled("RB SPACE INFO\n";underline=true)
  print(io,"Reduced basis space with #(basis space, basis time) = ($nbs,$nbt)\n")
end

get_basis_space(rb::RBSpace) = rb.basis_space
get_basis_time(rb::RBSpace) = rb.basis_time
get_space_ndofs(rb::RBSpace) = size(rb.basis_space,1)
get_time_ndofs(rb::RBSpace) = size(rb.basis_time,1)
get_rb_space_ndofs(rb::RBSpace) = size(rb.basis_space,2)
get_rb_time_ndofs(rb::RBSpace) = size(rb.basis_time,2)
get_rb_ndofs(rb::RBSpace) = get_rb_time_ndofs(rb)*get_rb_space_ndofs(rb)

function num_rb_dofs(rb::RBSpace)
  size(rb.basis_space,2)*size(rb.basis_time,2)
end

function save(info::RBInfo,rb::RBSpace)
  path = joinpath(info.rb_path,"rb")
  save(path,rb)
end

function load(info::RBInfo,T::Type{RBSpace})
  path = joinpath(info.rb_path,"rb")
  load(path,T)
end

function reduced_basis(info::RBInfo,feop::PTFEOperator,snaps::Snapshots;kwargs...)
  norm_style = info.norm_style
  norm_matrix = get_norm_matrix(info,feop;norm_style)
  reduced_basis(info,feop,snaps,norm_matrix;kwargs...)
end

function reduced_basis(
  info::RBInfo,
  feop::PTFEOperator,
  s::Snapshots,
  norm_matrix;
  nsnaps_state=50)

  _s = Snapshots(map(x->x[1:nsnaps_state],s.snaps))
  basis_space_nnz,basis_time = compress(info,feop,_s,norm_matrix)
  basis_space = recast(basis_space_nnz)
  rbspace = RBSpace(basis_space,basis_time)
  show(rbspace)
  return rbspace
end

function recast(x::AbstractVector,rb::RBSpace)
  basis_space = get_basis_space(rb)
  basis_time = get_basis_time(rb)
  ns_rb = get_rb_space_ndofs(rb)
  nt_rb = get_rb_time_ndofs(rb)

  x_mat = reshape(x,nt_rb,ns_rb)
  xrb_mat = basis_space*(basis_time*x_mat)'
  xrb = collect(eachcol(xrb_mat))
  return xrb
end

function recast(x::PTArray,rb::RBSpace{T}) where T
  time_ndofs = get_time_ndofs(rb)
  nparams = length(x)
  array = Vector{Vector{T}}(undef,time_ndofs*nparams)
  @inbounds for i = 1:nparams
    array[(i-1)*time_ndofs+1:i*time_ndofs] = recast(x[i],rb)
  end
  PTArray(array)
end

function space_time_projection(x::PTArray,rb::RBSpace{T}) where T
  time_ndofs = get_time_ndofs(rb)
  nparams = Int(length(x)/time_ndofs)

  array = Vector{Vector{T}}(undef,nparams)
  @inbounds for np = 1:nparams
    x_np = hcat(x[(np-1)*time_ndofs+1:np*time_ndofs]...)
    array[np] = space_time_projection(x_np,rb)
  end

  return PTArray(array)
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
  s_proj_mat = hcat([x[:] for x in s_proj]...)'  # time_ndofs x ns_row*ns_col
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

function test_reduced_basis(mat::AbstractMatrix,rb::RBSpace)
  rb_proj = space_time_projection(mat,rb)
  rb_approx = recast(rb_proj,rb)
  err = maximum(abs.(mat-rb_approx))
  println("RB approximation error in infty norm of snapshot $n = $err")
  return rb_proj
end
