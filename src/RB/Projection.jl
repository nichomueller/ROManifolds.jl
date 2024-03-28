abstract type Projection end

get_basis_space(a::Projection) = @abstractmethod
get_basis_time(a::Projection) = @abstractmethod
get_basis_spacetime(a::Projection) = @abstractmethod
num_reduced_space_dofs(a::Projection) = @abstractmethod
num_reduced_times(a::Projection) = @abstractmethod
num_fe_dofs(a::Projection) = @abstractmethod
num_reduced_dofs(a::Projection) = @abstractmethod

function Projection(s::AbstractSnapshots,args...;kwargs...)
  if num_space_dofs(s) < num_times(s)
    compute_bases_time_space(s,args...;kwargs...)
  else
    compute_bases_space_time(s,args...;kwargs...)
  end
end

function Projection(s::NnzSnapshots,args...;kwargs...)
  basis = compute_bases_space_time(s,args...;kwargs...)
  recast_basis(s,basis)
end

function compute_bases_space_time(s::AbstractSnapshots,norm_matrix=nothing;kwargs...)
  basis_space = tpod(s,norm_matrix;kwargs...)
  compressed_s2 = compress(s,basis_space,norm_matrix) |> change_mode
  basis_time = tpod(compressed_s2;kwargs...)
  PODBasis(basis_space,basis_time)
end

function compute_bases_time_space(s::AbstractSnapshots,norm_matrix=nothing;kwargs...)
  s2 = change_mode(s)
  basis_time = tpod(s2;kwargs...)
  compressed_s = compress(s2,basis_time) |> change_mode
  basis_space = tpod(compressed_s,norm_matrix;kwargs...)
  PODBasis(basis_space,basis_time)
end

struct PODBasis{A,B,C} <: Projection
  basis_space::A
  basis_time::B
  metadata::C
end

function PODBasis(basis_space,basis_time)
  metadata = nothing
  PODBasis(basis_space,basis_time,metadata)
end

get_basis_space(b::PODBasis) = b.basis_space
get_basis_time(b::PODBasis) = b.basis_time
num_space_dofs(b::PODBasis) = size(get_basis_space(b),1)
FEM.num_times(b::PODBasis) = size(get_basis_time(b),1)
num_reduced_space_dofs(b::PODBasis) = size(get_basis_space(b),2)
num_reduced_times(b::PODBasis) = size(get_basis_time(b),2)
num_fe_dofs(a::PODBasis) = num_space_dofs(a)*num_times(a)
num_reduced_dofs(a::PODBasis) = num_reduced_space_dofs(a)*num_reduced_times(a)

function recast_basis(s::NnzSnapshots,b::PODBasis)
  basis_space = recast(s,get_basis_space(b))
  basis_time = get_basis_time(b)
  PODBasis(basis_space,basis_time)
end

function recast(x::Vector,b::PODBasis)
  basis_space = get_basis_space(b)
  basis_time = get_basis_time(b)
  ns = num_reduced_space_dofs(b)
  nt = num_reduced_times(b)

  X = reshape(x,ns,nt)
  (basis_space*X)*basis_time'
end

# TT interface

function Projection(s::TTSnapshots,args...;kwargs...)
  cores = ttsvd(s,args...;kwargs...)
  TTSVDCores(cores)
end

function Projection(s::NnzTTSnapshots,args...;kwargs...)
  cores = ttsvd(s,args...;kwargs...)
  basis = TTSVDCores(cores)
  recast_basis(s,basis)
end

# D ∈ {2,3,4}: space (x,y,z) + time
struct TTSVDCores{D,A,B} <: Projection
  cores::A
  basis_spacetime::B
  function TTSVDCores(
    cores::A,
    basis_spacetime::B=cores2matrix(cores)
    ) where {A,B}

    D = length(cores)
    new{D,A,B}(cores,basis_spacetime)
  end
end

get_basis_space(b::TTSVDCores) = cores2matrix(b.cores[1:end-1])
get_basis_time(b::TTSVDCores) = cores2matrix(b.cores[end])
get_basis_spacetime(b::TTSVDCores) = b.basis_spacetime
num_space_dofs(b::TTSVDCores) = prod(size.(b.cores[1:end-1],2))
num_space_dofs(b::TTSVDCores,k::Integer) = size(b.cores[k],2)
FEM.num_times(b::TTSVDCores) = size(b.cores[end],2)
num_reduced_space_dofs(b::TTSVDCores) = prod(size.(b.cores[1:end-1],3))
num_reduced_space_dofs(b::TTSVDCores,k::Integer) = size(b.cores[k],3)
num_reduced_times(b::TTSVDCores) = size(b.cores[end],3)
num_fe_dofs(b::TTSVDCores) = num_space_dofs(b)*num_times(b)
num_reduced_dofs(b::TTSVDCores) = num_reduced_times(b)

function _cores2matrix(a::AbstractArray{T,3},b::AbstractArray{T,3}) where T
  nrows = size(a,2)*size(b,2)
  ncols = size(b,3)
  c = zeros(T,nrows,ncols)
  for j = 1:ncols
    for α = axes(a,3)
      @inbounds @views c[:,j] += kronecker(b[α,:,j],a[1,:,α])
    end
  end
  return c
end

function _cores2matrix(a::AbstractArray{T,3},b::AbstractArray{T,3}...) where T
  _cores2matrix(_cores2matrix(a,b[1]),b[2:end]...)
end

function cores2matrix(cores::Vector{<:AbstractArray{T,3}}) where T
  _cores2matrix(cores[1],cores[2:end]...)
end

function cores2matrix(core::AbstractArray{T,3}) where T
  pcore = permutedims(core,(2,1,3))
  return reshape(pcore,size(pcore,1),:)
end

function recast_basis(s::NnzTTSnapshots,b::TTSVDCores)
  basis_spacetime = recast(s,get_basis_spacetime(b))
  TTSVDCores(b.cores,basis_spacetime)
end

function recast(x::TTVector,b::TTSVDCores)
  basis_spacetime = b.basis_spacetime
  Ns = num_space_dofs(b)
  Nt = num_times(b)

  xrec = basis_spacetime*x
  reshape(xrec,Ns,Nt)
end

# multi field interface

struct BlockProjection{A,N}
  array::Array{A,N}
  touched::Array{Bool,N}
  function BlockProjection(
    array::Array{A,N},
    touched::Array{Bool,N}
    ) where {A<:Projection,N}

    @check size(array) == size(touched)
    new{A,N}(array,touched)
  end
end

function BlockProjection(k::BlockMap{N},a::A...) where {A<:Projection,N}
  array = Array{A,N}(undef,k.size)
  touched = fill(false,k.size)
  for (t,i) in enumerate(k.indices)
    array[i] = a[t]
    touched[i] = true
  end
  BlockProjection(array,touched)
end

Base.size(b::BlockProjection,i...) = size(b.array,i...)
Base.length(b::BlockProjection) = length(b.array)
Base.eltype(::Type{<:BlockProjection{A}}) where A = A
Base.eltype(b::BlockProjection{A}) where A = A
Base.ndims(b::BlockProjection{A,N}) where {A,N} = N
Base.ndims(::Type{BlockProjection{A,N}}) where {A,N} = N
Base.copy(b::BlockProjection) = BlockProjection(copy(b.array),copy(b.touched))
Base.eachindex(b::BlockProjection) = eachindex(b.array)
function Base.getindex(b::BlockProjection,i...)
  if !b.touched[i...]
    return nothing
  end
  b.array[i...]
end
function Base.setindex!(b::BlockProjection,v,i...)
  @check b.touched[i...] "Only touched entries can be set"
  b.array[i...] = v
end

function get_touched_blocks(b::BlockProjection)
  findall(b.touched)
end

function get_basis_space(b::BlockProjection{A,N}) where {A,N}
  basis_space = Array{Matrix{Float64},N}(undef,size(b))
  touched = b.touched
  for i in eachindex(b)
    if touched[i]
      basis_space[i] = get_basis_space(b[i])
    end
  end
  return ArrayBlock(basis_space,b.touched)
end

function get_basis_time(b::BlockProjection{A,N}) where {A,N}
  basis_time = Array{Matrix{Float64},N}(undef,size(b))
  touched = b.touched
  for i in eachindex(b)
    if touched[i]
      basis_time[i] = get_basis_time(b[i])
    end
  end
  return ArrayBlock(basis_time,b.touched)
end

function num_space_dofs(b::BlockProjection)
  dofs = zeros(Int,length(b))
  for i in eachindex(b)
    if b.touched[i]
      dofs[i] = num_space_dofs(b[i])
    end
  end
  return dofs
end

function num_reduced_space_dofs(b::BlockProjection)
  dofs = zeros(Int,length(b))
  for i in eachindex(b)
    if b.touched[i]
      dofs[i] = num_reduced_space_dofs(b[i])
    end
  end
  return dofs
end

FEM.num_times(b::BlockProjection) = num_times(b[findfirst(b.touched)])

function num_reduced_times(b::BlockProjection)
  dofs = zeros(Int,length(b))
  for i in eachindex(b)
    if b.touched[i]
      dofs[i] = num_reduced_times(b[i])
    end
  end
  return dofs
end

function num_fe_dofs(b::BlockProjection)
  ndofs = 0
  for i in eachindex(b)
    if b.touched[i]
      ndofs += num_fe_dofs(b[i])
    end
  end
  return ndofs
end

function num_reduced_dofs(b::BlockProjection)
  ndofs = 0
  for i in eachindex(b)
    if b.touched[i]
      ndofs += num_reduced_dofs(b[i])
    end
  end
  return ndofs
end

function Projection(s::BlockSnapshots;kwargs...)
  norm_matrix = fill(nothing,size(s))
  reduced_basis(s,norm_matrix;kwargs...)
end

function Projection(s::BlockSnapshots,norm_matrix;kwargs...)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  bases = Any[reduced_basis(s[i],norm_matrix[Block(i,i)];kwargs...) for i in active_block_ids]
  BlockProjection(block_map,bases...)
end

function enrich_basis(b::BlockProjection{<:PODBasis},norm_matrix::BlockMatrix,supr_op::BlockMatrix)
  basis_space = add_space_supremizers(get_basis_space(b),norm_matrix,supr_op)
  basis_time = add_time_supremizers(get_basis_time(b))
  basis = BlockProjection(map(PODBasis,basis_space,basis_time),b.touched)
  return basis
end

function add_space_supremizers(basis_space,norm_matrix::BlockMatrix,supr_op::BlockMatrix)
  basis_primal,basis_dual... = basis_space.array
  A = norm_matrix[Block(1,1)]
  Chol = cholesky(A)
  for i = eachindex(basis_dual)
    b_i = supr_op[Block(1,i+1)] * basis_dual[i]
    supr_i = Chol \ b_i
    gram_schmidt!(supr_i,basis_primal,A)
    basis_primal = hcat(basis_primal,supr_i)
  end
  return [basis_primal,basis_dual...]
end

function add_time_supremizers(basis_time;kwargs...)
  basis_primal,basis_dual... = basis_time.array
  for i = eachindex(basis_dual)
    basis_primal = add_time_supremizers(basis_primal,basis_dual[i];kwargs...)
  end
  return [basis_primal,basis_dual...]
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
  ntd_minus_ntp = size(basis_dual,2) - size(basis_primal,2)
  if ntd_minus_ntp > 0
    for ntd = 1:ntd_minus_ntp
      basis_primal,basis_pd = enrich(basis_primal,basis_pd,basis_dual[:,ntd])
      count += 1
    end
  end

  ntd = 1
  while ntd ≤ size(basis_pd,2)
    proj = ntd == 1 ? zeros(size(basis_pd,1)) : orth_projection(basis_pd[:,ntd],basis_pd[:,1:ntd-1])
    dist = norm(basis_pd[:,1]-proj)
    if dist ≤ tol
      basis_primal,basis_pd = enrich(basis_primal,basis_pd,basis_dual[:,ntd])
      count += 1
      ntd = 0
    else
      basis_pd[:,ntd] .-= proj
    end
    ntd += 1
  end

  return basis_primal
end
