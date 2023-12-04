struct Snapshots{T<:AbstractArray}
  snaps::Vector{PTArray{T}}
end

Base.length(s::Snapshots) = length(s.snaps)
Base.size(s::Snapshots,args...) = size(testitem(first(s.snaps)),args...)
Base.eachindex(s::Snapshots) = eachindex(s.snaps)
Base.lastindex(s::Snapshots) = num_params(s)
Base.copy(s::Snapshots) = Snapshots(copy.(s.snaps))
num_space_dofs(s::Snapshots) = size(s,1)
FEM.num_time_dofs(s::Snapshots) = length(s)
num_params(s::Snapshots) = length(first(s.snaps))

function Base.getindex(s::Snapshots{T},idx) where T
  time_ndofs = num_time_dofs(s)
  nrange = length(idx)
  array = Vector{T}(undef,time_ndofs*nrange)
  for (i,r) in enumerate(idx)
    for nt in 1:time_ndofs
      array[(i-1)*time_ndofs+nt] = s.snaps[nt][r]
    end
  end
  return PTArray(array)
end

function Base.vcat(s::Snapshots{T}...) where T
  l = length(first(s))
  vsnaps = Vector{PTArray{T}}(undef,l)
  @inbounds for i = 1:l
    vsnaps[i] = vcat(map(n->s[n].snaps[i],eachindex(s))...)
  end
  Snapshots(vsnaps)
end

function Utils.save(rbinfo::RBInfo,s::Snapshots)
  path = joinpath(rbinfo.fe_path,"fesnaps")
  save(path,s)
end

function Utils.load(rbinfo::RBInfo,T::Type{Snapshots{S}}) where S
  path = joinpath(rbinfo.fe_path,"fesnaps")
  load(path,T)
end

function collect_solutions(
  rbinfo::RBInfo,
  fesolver::PODESolver,
  feop::PTFEOperator)

  uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
  ode_op = get_algebraic_operator(feop)
  nparams = rbinfo.nsnaps_state+rbinfo.nsnaps_test
  params = realization(feop,nparams)
  u0 = get_free_dof_values(uh0(params))
  time_ndofs = num_time_dofs(fesolver)
  T = get_vector_type(feop.test)
  uμt = PODESolution(fesolver,ode_op,params,u0,t0,tf)
  snaps = Vector{PTArray{T}}(undef,time_ndofs)
  println("Computing fe solution: time marching across $time_ndofs instants, for $nparams parameters")
  stats = @timed for (snap,n) in uμt
    snaps[n] = copy(snap)
  end
  println("Time marching complete")
  sols = Snapshots(snaps)
  cinfo = ComputationInfo(stats,nparams)
  return sols,params,cinfo
end
