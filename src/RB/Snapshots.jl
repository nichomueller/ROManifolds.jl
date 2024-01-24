struct Snapshots{A<:AbstractVector{<:ParamArray}}
  snaps::A
end

Base.eltype(::Snapshots{T}) where T = eltype(eltype(T))
Base.length(s::Snapshots) = length(s.snaps)
Base.size(s::Snapshots,args...) = size(testitem(first(s.snaps)),args...)
Base.eachindex(s::Snapshots) = eachindex(s.snaps)
Base.lastindex(s::Snapshots) = num_params(s)
Base.copy(s::Snapshots) = Snapshots(copy.(s.snaps))
num_space_dofs(s::Snapshots) = size(s,1)
FEM.num_time_dofs(s::Snapshots) = length(s)
num_params(s::Snapshots) = length(first(s.snaps))

function Base.getindex(s::Snapshots,idx)
  time_ndofs = num_time_dofs(s)
  nrange = length(idx)
  array = Vector{Vector{eltype(s)}}(undef,time_ndofs*nrange)
  for (i,r) in enumerate(idx)
    for nt in 1:time_ndofs
      array[(i-1)*time_ndofs+nt] = s.snaps[nt][r]
    end
  end
  return ParamArray(array)
end

function Base.vcat(s::Snapshots...)
  vsnaps = map(eachindex(first(s))) do i
    vcat(map(x->x.snaps[i],s)...)
  end
  Snapshots(vsnaps)
end

function Utils.save(rbinfo::RBInfo,s::Snapshots)
  path = joinpath(rbinfo.fe_path,"fesnaps")
  save(path,s)
end

function Utils.load(rbinfo::RBInfo,::Type{Snapshots{T}}) where T
  path = joinpath(rbinfo.fe_path,"fesnaps")
  load(path,Snapshots{AbstractVector{<:ParamArray{T}}})
end

function collect_solutions(
  rbinfo::RBInfo,
  fesolver::ODESolver,
  feop::TransientPFEOperator)

  nparams = rbinfo.nsnaps_state+rbinfo.nsnaps_test
  params = realization(feop,nparams)
  u0 = get_free_dof_values(fesolver.uh0(params))
  time_ndofs = num_time_dofs(fesolver)
  uμt = ODEPSolution(fesolver,feop,params,u0,fesolver.t0,fesolver.tf)
  println("Computing fe solution: time marching across $time_ndofs instants, for $nparams parameters")
  stats = @timed begin
    snaps = map(uμt) do (snap,n)
      copy(snap)
    end
  end
  println("Time marching complete")
  sols = Snapshots(snaps)
  cinfo = ComputationInfo(stats,nparams)
  return sols,params,cinfo
end
