struct Snapshots{T}
  snaps::Vector{PTArray{NnzArray{T}}}
  function Snapshots(snaps::AbstractVector{<:PTArray{T}}) where T
    snnz = compress(snaps)
    new{T}(snnz)
  end
end

const SingleFieldSnapshots{T} = Snapshots{T} where {T<:AbstractArray}
const MultiFieldSnapshots{T} = Snapshots{T} where {T<:BlockArray}

num_time_steps(s::Snapshots) = length(s.snaps)
num_params(s::Snapshots) = length(testitem(s))
Base.length(s::Snapshots) = num_params(s)
Arrays.testitem(s::Snapshots) = testitem(s.snaps)

function Base.size(s::Snapshots)
  s1 = testitem(s)
  s11 = testitem(s1)
  size(s11,1),size(s11,2)*length(s)
end

Base.size(s::Snapshots,i::Int) = size(s)[i]

function Base.show(io::IO,s::Snapshots{T}) where T
  nsnaps = num_params(s)
  print(io,"Structure storing $nsnaps snapshots of type $T")
end

Base.length(s::Snapshots) = length(s.snaps)
Base.eltype(::Snapshots{T}) where T = eltype(T)
Base.eachindex(s::Snapshots) = eachindex(s.snaps)

function Base.getindex(s::Snapshots,i...)
  s.snaps[i...]
end

function Base.setindex!(s::Snapshots,v,i...)
  s.snaps[i...] = v
end

Base.copy(s::Snapshots) = Snapshots(copy(s.snaps))

function Base.fill!(s::Snapshots,v)
  for n in 1:num_time_steps(s)
    fill!(s.snaps[n],v)
  end
end

Base.collect(s::Snapshots;transpose=false) = _collect_snaps(Val(transpose),s)

function _collect_snaps(::Val{false},s::Snapshots{T}) where T
  # map(x->get_arrays(x),s.snaps...)
  snaps = map(get_nonzero_val,get_arrays(s.snaps...)...)
  sz = size(snaps)
  S = zeros(T,sz...)
  @inbounds for i = 1:sz[1]
    for j = 1:sz[2]
      S[:,j] =
    end
  end
  return hcat(get_arrays(snaps)...)
end

function _collect_snaps(::Val{true},s::Snapshots{T}) where T
  snaps = get_snaps(s)
  snaps_t = map(transpose,snaps)
  return hcat(get_arrays(snaps_t)...)
end

function compress_snapshots(s::Snapshots,args...;kwargs...)
  issteady = num_time_steps(s) == 1
  compress_snapshots(Val(issteady),s,args...;kwargs...)
end

function compress_snapshots(::Val{false},s::Snapshots,args...;kwargs...)
  transpose = size(s,1) < size(s,1)
  snaps = collect(s;transpose)
  nsnaps = length(s)
  b1 = tpod(snaps,args...;kwargs...)
  compressed_b1 = prod(b1,snaps)
  compressed_snaps_t = change_mode(compressed_b1,nsnaps)
  b2 = tpod(compressed_snaps_t;kwargs...)
  if size(s,1) < size(s,1)
    return b1,b2
  else
    return b2,b1
  end
end

function compress_snapshots(::Val{true},s::Snapshots,args...;kwargs...)
  snaps = collect(s)
  basis_space = tpod(snaps,args...;kwargs...)
  basis_time = ones(eltype(s),1,1)
  basis_space,basis_time
end

function collect_solutions(
  op::PTFEOperator,
  solver::ThetaMethod,
  μ::Table,
  u0::PTCellField)

  ode_op = get_algebraic_operator(op)
  uμt = PODESolution(solver,ode_op,μ,u0,t0,tF)
  solutions = PTArray[]
  for (u,t) in uμt
    printstyled("Computing fe solution at time $t for every parameter\n";color=:blue)
    push!(solutions,u)
  end
  return solutions
end

# function collect_residuals(
#   feop::PTFEOperator,
#   fesolver::ODESolver,
#   snaps::Snapshots,
#   params::Table,
#   trian::Triangulation,
#   args...;
#   kwargs...)

#   printstyled("Generating residuals snapshots\n";color=:blue)
#   sols = get_snaps(snaps)
#   aff,ress = collect_residuals(feop,fesolver,sols,params,trian,args...;kwargs...)

#   Snapshots(aff,ress)
# end

# function collect_jacobians(
#   feop::PTFEOperator,
#   fesolver::ODESolver,
#   snaps::Snapshots,
#   params::Table,
#   trian::Triangulation,
#   args...;
#   kwargs...)

#   printstyled("Generating jacobians snapshots\n";color=:blue)
#   sols = get_snaps(snaps)
#   aff,jacs = collect_jacobians(feop,fesolver,sols,params,trian,args...;kwargs...)

#   Snapshots(aff,jacs)
# end

function save(info::RBInfo,s::Snapshots)
  if info.save_structures
    path = joinpath(info.fe_path,"fesnaps")
    save(path,s)
  end
end

function save(info::RBInfo,params::Table)
  if info.save_structures
    path = joinpath(info.fe_path,"params")
    save(path,params)
  end
end

function load(T::Type{Snapshots},info::RBInfo)
  path = joinpath(info.fe_path,"fesnaps")
  s = load(T,path)
  s
end

function load(T::Type{Table},info::RBInfo)
  path = joinpath(info.fe_path,"params")
  load(T,path)
end
