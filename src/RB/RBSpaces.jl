struct RBSpace{T}
  basis_space::Matrix{T}
  basis_time::Matrix{T}

  function RBSpace(
    basis_space::Matrix{T},
    basis_time::Matrix{T}) where T
    new{T}(basis_space,basis_time)
  end
end

get_basis_space(rb::RBSpace) = rb.basis_space
get_basis_time(rb::RBSpace) = rb.basis_time

function num_rb_dofs(rb::RBSpace)
  size(rb.basis_space,2)*size(rb.basis_time,2)
end

function Algebra.allocate_vector(rb_row::RBSpace{T}) where T
  zeros(T,num_rb_dofs(rb_row))
end

function Algebra.allocate_matrix(rb_row::RBSpace{T},rb_col::RBSpace{T}) where T
  zeros(T,num_rb_dofs(rb_row),num_rb_dofs(rb_col))
end

function save(info::RBInfo,rb::RBSpace)
  if info.save_structures
    path = joinpath(info.rb_path,"rb")
    save(path,rb)
  end
end

function load(info::RBInfo,T::Type{RBSpace})
  path = joinpath(info.rb_path,"rb")
  load(path,T)
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

function compress(info::RBInfo,::PTFEOperator,snaps,args...)
  nzm = NnzArray(snaps)
  ϵ = info.ϵ
  steady = num_time_dofs(nzm) == 1 ? SteadyPOD() : DefaultPOD()
  compress(nzm,steady,args...;ϵ)
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
  return vec(st_proj')
end

function space_time_projection(
  mat::NnzMatrix{T},rb_row::RBSpace,rb_col::RBSpace;combine_projections=(x,y)->x) where T
  basis_space_row = get_basis_space(rb_row)
  basis_time_row = get_basis_time(rb_row)
  basis_space_col = get_basis_space(rb_col)
  basis_time_col = get_basis_time(rb_col)
  ns_row,ns_col = size(basis_space_row,2),size(basis_space_col,2)
  nt_row,nt_col = size(basis_time_row,2),size(basis_time_col,2)

  s_proj = compress(basis_space_row,basis_space_col,mat)
  s_proj_mat = hcat([x[:] for x in s_proj]...)'  # time_ndofs x ns_row*ns_col
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
