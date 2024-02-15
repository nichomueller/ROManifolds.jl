function RB.get_norm_matrix(
  info::RBInfo,
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace)

  map(local_views(trial),local_views(test)) do trial,test
    RB.get_norm_matrix(info,trial,test)
  end
end

struct DistributedTransientSnapshots{T<:PVector{<:AbstractTransientSnapshots}}
  snaps::T
end

function DistributedSnapshots(snaps::PVector{<:AbstractTransientSnapshots})
  DistributedTransientSnapshots(snaps)
end

GridapDistributed.local_views(s::DistributedTransientSnapshots) = local_views(s.snaps)

function RB.Snapshots(
  values::AbstractVector{<:PVector{V}},
  args...) where V

  index_partition = first(values).index_partition
  parts = map(part_id,index_partition)
  snaps = map(parts) do part
    vals_part = Vector{V}(undef,length(values))
    for (k,v) in enumerate(values)
      map(local_views(v),index_partition) do val,ip
        if part_id(ip) == part
          vals_part[k] = val
        end
      end
    end
    Snapshots(vals_part,args...)
  end
  psnaps = PVector(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

function RB.get_values(s::DistributedTransientSnapshots)
  snaps = map(local_views(s)) do s
    get_values(s)
  end
  index_partition = s.snaps.index_partition
  PVector(snaps,index_partition)
end

function RB.get_realization(s::DistributedTransientSnapshots)
  s1 = PartitionedArrays.getany(local_views(s.snaps))
  get_realization(s1)
end

function RB.select_snapshots(s::DistributedTransientSnapshots,args...;kwargs...)
  snaps = map(local_views(s)) do s
    select_snapshots(s,args...;kwargs...)
  end
  index_partition = s.snaps.index_partition
  psnaps = PVector(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

function RB.reverse_snapshots(s::DistributedTransientSnapshots)
  snaps = map(local_views(s)) do s
    reverse_snapshots(s)
  end
  index_partition = s.snaps.index_partition
  psnaps = PVector(snaps,index_partition)
  DistributedSnapshots(psnaps)
end

struct DistributedRBSpace{T<:AbstractVector{<:RBSpace}} <: DistributedFESpace
  spaces::T
end

GridapDistributed.local_views(a::DistributedRBSpace) = a.spaces

function Arrays.evaluate(
  U::DistributedRBSpace{<:AbstractVector{<:TrialRBSpace}},args...)
  spaces = map(U->evaluate(U,args...),local_views(U))
  DistributedRBSpace(spaces)
end

(U::DistributedRBSpace)(r) = evaluate(U,r)
(U::DistributedRBSpace)(μ,t) = evaluate(U,μ,t)

function ODETools.∂t(U::DistributedRBSpace{<:AbstractVector{<:TrialRBSpace}})
  spaces = map(U->∂t(U),local_views(U))
  DistributedRBSpace(spaces)
end

function ODETools.∂tt(U::DistributedRBSpace{<:AbstractVector{<:TrialRBSpace}})
  spaces = map(U->∂tt(U),local_views(U))
  DistributedRBSpace(spaces)
end

function RB.reduced_fe_space(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s::DistributedTransientSnapshots)

  trial = get_trial(feop)
  dtrial = _to_distributed_fe_space(trial)
  test = get_test(feop)
  norm_matrix = RB.get_norm_matrix(info,feop)
  reduced_trial,reduced_test = map(
    local_views(dtrial),
    local_views(test),
    local_views(s),
    local_views(norm_matrix)
    ) do trial,test,s,norm_matrix

    soff = select_snapshots(s,RB.offline_params(info))
    basis_space,basis_time = reduced_basis(soff,norm_matrix;ϵ=RB.get_tol(info))
    reduced_trial = TrialRBSpace(trial,basis_space,basis_time)
    reduced_test = TestRBSpace(test,basis_space,basis_time)
    reduced_trial,reduced_test
  end |> tuple_of_arrays

  dtrial = DistributedRBSpace(reduced_trial)
  dtest = DistributedRBSpace(reduced_test)
  return dtrial,dtest
end

function GridapDistributed._find_vector_type(
  spaces::AbstractVector{<:RBSpace},gids)
  T = get_vector_type(PartitionedArrays.getany(spaces))
  if isa(gids,PRange)
    vector_type = typeof(PVector{T}(undef,partition(gids)))
  else
    vector_type = typeof(BlockPVector{T}(undef,gids))
  end
  return vector_type
end

function RB.compress(r::DistributedRBSpace,s::DistributedTransientSnapshots)
  partition = s.snaps.index_partition
  vector_partition = map(local_views(r),local_views(s)) do r,s
    compress(r,s)
  end
  PVector(vector_partition,partition)
end

function RB.compress(
  trial::DistributedRBSpace,
  test::DistributedRBSpace,
  xmat::ParamVector;
  kwargs...)

  partition = xmat.index_partition
  vector_partition = map(local_views(trial),local_views(test),local_views(xmat)
    ) do trial,test,xmat
    compress(trial,test,xmat;kwargs...)
  end
  PVector(vector_partition,partition)
end

function RB.recast(r::DistributedRBSpace,red_x::PVector)
  partition = red_x.index_partition
  vector_partition = map(local_views(r),local_views(red_x)) do r,red_x
    recast(r,red_x)
  end
  PVector(vector_partition,partition)
end

struct DistributedContribution{T<:AbstractVector{<:Contribution}} <: FEM.AbstractContribution
  contribs::T
end

GridapDistributed.local_views(c::DistributedContribution) = c.contributions

CellData.get_domains(c::DistributedContribution) = map(get_domains,local_values(c))
FEM.get_values(c::DistributedContribution) = map(get_values,local_values(c))

function CellData.get_contribution(c::DistributedContribution,trian::DistributedTriangulation)
  map(local_views(c),local_views(trian)) do c,trian
    get_contribution(c,trian)
  end
end

function CellData.add_contribution!(c::DistributedContribution,b,trian::DistributedTriangulation)
  map(local_views(c),local_views(b),local_views(trian)) do c,b,trian
    add_contribution!(c,b,trian)
  end
end

const DistributedArrayContribution = DistributedContribution{T} where {T<:AbstractVector{<:ArrayContribution}}

parray_contribution() =

const DistributedAffineContribution = DistributedContribution{T} where {T<:AbstractVector{<:AffineContribution}}

function RB.reduced_vector_form(
  solver::RBSolver,
  op::RBOperator,
  c::DistributedArrayContribution)

  a = map(local_views(c)) do c
    reduced_vector_form(solver,op,c)
  end
  DistributedContribution(a)
end

function RB.reduced_matrix_form(
  solver::RBSolver,
  op::RBOperator,
  c::DistributedArrayContribution;
  kwargs...)

  a = map(local_views(c)) do c
    reduced_matrix_form(solver,op,c;kwargs...)
  end
  DistributedContribution(a)
end

# post process

get_ind_part_filename(info::RBInfo) = info.dir * "/index_partition.jld"

function get_dir_part(dir::AbstractString,part::Integer)
  dir_part = joinpath(dir,"part_$(part)")
  FEM.create_dir(dir_part)
  dir_part
end

function get_part_filename(filename::AbstractString,part::Integer)
  _filename,extension = splitext(filename)
  dir,varname = splitdir(_filename)
  dir_part = get_dir_part(dir,part)
  joinpath(dir_part,varname*extension)
end

function DrWatson.save(info::RBInfo,s::DistributedTransientSnapshots)
  i_filename = get_ind_part_filename(info)
  s_filename = RB.get_snapshots_filename(info)
  index_partition = s.snaps.index_partition
  map(local_views(s),local_views(index_partition)) do s,index_partition
    part = part_id(index_partition)
    i_part_filename = get_part_filename(i_filename,part)
    s_part_filename = get_part_filename(s_filename,part)
    serialize(i_part_filename,index_partition)
    serialize(s_part_filename,s)
  end
end

function load_distributed_snapshots(distribute,info::RBInfo)
  i_filename = get_ind_part_filename(info)
  s_filename = RB.get_snapshots_filename(info)
  i_parts,s_parts = map(readdir(info.dir;join=true)) do dir
    part = parse(Int,dir[end])
    i_part_filename = get_part_filename(i_filename,part)
    s_part_filename = get_part_filename(s_filename,part)
    deserialize(i_part_filename),deserialize(s_part_filename)
  end |> tuple_of_arrays
  index_partition = distribute(i_parts)
  snaps_partition = distribute(s_parts)
  psnaps = PVector(snaps_partition,index_partition)
  DistributedSnapshots(psnaps)
end
