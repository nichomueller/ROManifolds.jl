struct OldTTSVDCores{D,A,B} <: Projection
  cores::A
  basis_spacetime::B
  function OldTTSVDCores(
    cores::A,
    basis_spacetime::B=cores2basis(cores...)
    ) where {A,B}

    D = length(cores)
    new{D,A,B}(cores,basis_spacetime)
  end
end

get_basis_space(b::OldTTSVDCores) = cores2basis(b.cores[1:end-1]...)
get_basis_time(b::OldTTSVDCores) = cores2basis(b.cores[end])
get_basis_spacetime(b::OldTTSVDCores) = b.basis_spacetime
num_space_dofs(b::OldTTSVDCores) = prod(size.(b.cores[1:end-1],2))
FEM.num_times(b::OldTTSVDCores) = size(b.cores[end],2)
num_reduced_space_dofs(b::OldTTSVDCores) = size(b.cores[end-1],3)
num_reduced_times(b::OldTTSVDCores) = size(b.cores[end],3)
num_fe_dofs(b::OldTTSVDCores) = num_space_dofs(b)*num_times(b)
num_reduced_dofs(b::OldTTSVDCores) = num_reduced_times(b)

function temp_old_basis(r)
  i = get_dof_permutation(r.space)[:]
  invi = invperm(i)
  _old_core_space = get_basis_space(r.basis)[invi,:]
  old_core_space = reshape(_old_core_space,1,size(_old_core_space)...)
  OldTTSVDCores([old_core_space,r.basis.core_time])
end

function temp_reduced_form(
  solver::RBSolver,
  s::S,
  trian::T,
  trial,test;
  kwargs...) where {S,T}

  old_basis_trial = temp_old_basis(trial)
  old_basis_test = temp_old_basis(test)
  mdeim_style = solver.mdeim_style
  basis = temp_reduced_basis(s;ϵ=get_tol(solver))
  lu_interp,integration_domain = temp_mdeim(mdeim_style,basis)
  proj_basis = temp_reduce_operator(mdeim_style,basis,old_basis_trial,old_basis_test;kwargs...)
  red_trian = reduce_triangulation(trian,integration_domain,trial,test)
  coefficient = allocate_coefficient(solver,basis)
  result = allocate_result(solver,trial,test)
  ad = AffineDecomposition(proj_basis,lu_interp,integration_domain,coefficient,result)
  return ad,red_trian
end

function temp_reduced_jacobian(
  solver::RBSolver,
  op::RBOperator,
  c::ArrayContribution;
  kwargs...)

  a,trians = map(get_domains(c),get_values(c)) do trian,s
    sold = OldTTNnzSnapshots(s.values,s.realization)
    trial = get_trial(op)
    test = get_test(op)
    temp_reduced_form(solver,sold,trian,trial,test;kwargs...)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function temp_reduced_jacobian(
  solver::ThetaMethodRBSolver,
  op::RBOperator,
  contribs::Tuple{Vararg{Any}})

  fesolver = get_fe_solver(solver)
  θ = fesolver.θ
  a = ()
  for (i,c) in enumerate(contribs)
    combine = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*(x-y)
    a = (a...,temp_reduced_jacobian(solver,op,c;combine))
  end
  return a
end

function temp_recast_indices(A::AbstractArray,indices::AbstractVector)
  nonzero_indices = get_nonzero_indices(first(A.values))
  entire_indices = nonzero_indices[indices]
  return entire_indices
end

function temp_mdeim(mdeim_style::MDEIMStyle,b)
  basis_space = first(b.cores)
  basis_spacetime = get_basis_spacetime(b)
  indices_spacetime = get_mdeim_indices(basis_spacetime)
  indices_space = fast_index(indices_spacetime,num_space_dofs(b))
  indices_time = slow_index(indices_spacetime,num_space_dofs(b))
  lu_interp = lu(view(basis_spacetime,indices_spacetime,:))
  recast_indices_space = temp_recast_indices(basis_space,indices_space)
  integration_domain = ReducedIntegrationDomain(recast_indices_space,indices_time)
  return lu_interp,integration_domain
end

function temp_reduce_operator(
  mdeim_style::SpaceTimeMDEIM,
  b::OldTTSVDCores,
  b_trial::OldTTSVDCores,
  b_test::OldTTSVDCores;
  kwargs...)

  bs = first(b.cores)
  bt = get_basis_time(b)
  bs_trial = get_basis_space(b_trial)
  bt_trial = get_basis_time(b_trial)
  bs_test = get_basis_space(b_test)
  bt_test = get_basis_time(b_test)

  ns = num_reduced_space_dofs(b)
  ns_test = num_reduced_space_dofs(b_test)
  ns_trial = num_reduced_space_dofs(b_trial)

  T = eltype(first(bs))
  M = Matrix{T}
  b̂st = Vector{M}(undef,num_reduced_dofs(b))

  cache_t = zeros(T,num_reduced_dofs(b_test),num_reduced_dofs(b_trial))
  @inbounds for i = 1:num_reduced_dofs(b)
    b̂st[i] = copy(cache_t)
  end

  b̂s = map(x->bs_test'*x*bs_trial,bs.values)

  @inbounds for i = 1:num_reduced_dofs(b)
    bti = view(bt,:,(i-1)*ns+1:i*ns)
    for i_test = 1:num_reduced_dofs(b_test), i_trial = 1:num_reduced_dofs(b_trial)
      ids_i_test = (i_test-1)*ns_test+1:i_test*ns_test
      ids_i_trial = (i_trial-1)*ns_trial+1:i_trial*ns_trial
      bti_test = view(bt_test,:,ids_i_test)
      bti_trial = view(bt_trial,:,ids_i_trial)
      b̂ti = combine_basis_time(bti,bti_trial,bti_test;kwargs...)
      cache_t[i_test,i_trial] = sum(dot.(b̂s,b̂ti))
    end
    b̂st[i] .+= copy(cache_t)
  end

  return ReducedMatrixOperator(mdeim_style,b̂st)
end

struct VecOfSparseMat2Arr3{Tv,Ti,V} <: AbstractArray{Tv,3}
  values::V
  function VecOfSparseMat2Arr3(values::V) where {Tv,Ti,V<:AbstractVector{<:AbstractSparseMatrix{Tv,Ti}}}
    new{Tv,Ti,V}(values)
  end
end

FEM.get_values(s::VecOfSparseMat2Arr3) = s.values
Base.size(s::VecOfSparseMat2Arr3) = (1,nnz(first(s.values)),length(s.values))

function Base.getindex(s::VecOfSparseMat2Arr3,i::Integer,j,k::Integer)
  @check i == 1
  nonzeros(s.values[k])[j]
end

function Base.getindex(s::VecOfSparseMat2Arr3,i::Integer,j,k)
  @check i == 1
  view(s,i,j,k)
end

function FEM.get_nonzero_indices(s::VecOfSparseMat2Arr3)
  get_nonzero_indices(first(s.values))
end

# old snaps

struct OldTTNnzSnapshots{T,N,P,R} <: TTSnapshots{T,N}
  values::P
  realization::R
  function OldTTNnzSnapshots(values::P,realization::R) where {P<:ParamTTArray,R}
    T = eltype(P)
    N = 3
    new{T,N,P,R}(values,realization)
  end
end

num_space_dofs(s::OldTTNnzSnapshots) = nnz(first(s.values))

function Base.getindex(s::OldTTNnzSnapshots,i::CartesianIndex{3})
  getindex(s,i.I...)
end

function Base.getindex(s::OldTTNnzSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace]
end

function temp_ttsvd(mat::AbstractArray{T,N},args...;kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  mat_k = copy(mat)
  for k = 1:N-1
    mat_k = reshape(mat_k,ranks[k]*sizes[k],:)
    U,Σ,V = svd(mat_k)
    rank = truncation(Σ;kwargs...)
    ranks[k+1] = rank
    mat_k = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    cores[k] = reshape(U[:,1:rank],ranks[k],sizes[k],rank)
  end
  return cores
end

function temp_reduced_basis(a::OldTTNnzSnapshots,args...;kwargs...)
  cores = temp_ttsvd(a,args...;kwargs...)
  b = OldTTSVDCores(cores)
  _space_core,time_core = b.cores
  space_core = temp_recast(a,_space_core)
  OldTTSVDCores([space_core,time_core],b.basis_spacetime)
end

function temp_recast(s::OldTTNnzSnapshots,a::AbstractArray{T,3}) where T
  @check size(a,1) == 1
  v = first(s.values)
  i,j, = findnz(v)
  m,n = size(v)
  asparse = map(eachcol(dropdims(a;dims=1))) do v
    sparse(i,j,v,m,n)
  end
  return VecOfSparseMat2Arr3(asparse)
end

function select_snapshots_entries(s::OldTTNnzSnapshots,ispace,itime)
  index_map = get_index_map(s)
  cispace = map(i->findfirst(index_map.==i),ispace)
  _select_snapshots_entries(s,cispace,itime)
end
