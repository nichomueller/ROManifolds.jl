abstract type RBSpace{T} end
abstract type TransientRBSpace{T} end

struct SingleFieldRBSpace{T} <: RBSpace{T}
  basis_space::NnzArray{T}
end

struct MultiFieldRBSpace{T} <: RBSpace{T}
  basis_space::Vector{NnzArray{T}}
end

struct TransientSingleFieldRBSpace{T} <: TransientRBSpace{T}
  basis_space::NnzArray{T}
  basis_time::NnzArray{T}
end

struct TransientMultiFieldRBSpace{T} <: TransientRBSpace{T}
  basis_space::Vector{NnzArray{T}}
  basis_time::Vector{NnzArray{T}}
end

get_basis_space(rb::SingleFieldRBSpace) = rb.basis_space

get_basis_space(rb::TransientSingleFieldRBSpace) = rb.basis_space

get_basis_time(rb::TransientSingleFieldRBSpace) = rb.basis_time

Base.getindex(rb::MultiFieldRBSpace,i::Int) = get_single_field(rb,i)

Base.getindex(rb::TransientMultiFieldRBSpace,i::Int) = get_single_field(rb,i)

function get_single_field(
  rb::MultiFieldRBSpace{T},
  fieldid::Int) where T

  SingleFieldRBSpace{T}(rb.basis_space[fieldid])
end

function get_single_field(
  rb::TransientMultiFieldRBSpace{T},
  fieldid::Int) where T

  TransientSingleFieldRBSpace{T}(rb.basis_space[fieldid],rb.basis_time[fieldid])
end

function compress_solution(
  s::SingleFieldSnapshots{T},
  ::ParamFEOperator,
  ::FESolver;
  kwargs...) where T

  basis_space = tpod(s;kwargs...)
  SingleFieldRBSpace{T}(basis_space)
end

function compress_solution(
  s::MultiFieldSnapshots{T},
  feop::ParamFEOperator,
  fe_solver::FESolver;
  compute_supremizers=false,
  kwargs...) where T

  snaps = collect_single_fields(s)
  bases_space = map(snap -> tpod(snap;kwargs...),snaps)
  if compute_supremizers
    add_space_supremizers!(bases_space,feop,fe_solver,s)
  end
  MultiFieldRBSpace{T}(bases_space)
end

function compress_solution(
  s::SingleFieldSnapshots{T},
  ::ParamTransientFEOperator,
  ::ODESolver;
  kwargs...) where T

  basis_space,basis_time = transient_tpod(s;kwargs...)
  TransientSingleFieldRBSpace{T}(basis_space,basis_time)
end

function compress_solution(
  s::MultiFieldSnapshots{T},
  feop::ParamTransientFEOperator,
  fe_solver::ODESolver;
  compute_supremizers=false,
  ttol=1e-2,
  kwargs...) where T

  snaps = collect_single_fields(s)
  bases = map(snap -> transient_tpod(snap;kwargs...),snaps)
  bases_space,bases_time = first.(bases),last.(bases)
  if compute_supremizers
    add_space_supremizers!(bases_space,feop,fe_solver,s)
    add_time_supremizers!(bases_time;ttol)
  end
  TransientMultiFieldRBSpace{T}(bases_space,bases_time)
end

for (Top,Tslv) in zip((:ParamFEOperator,:ParamTransientFEOperator),(:FESolver,:ODESolver))
  @eval begin
    function add_space_supremizers!(
      bases_space::Vector{<:NnzArray},
      feop::$Top,
      solver::$Tslv,
      snaps::MultiFieldSnapshots;
      kwargs...)

      sbu,sbdual... = bases_space
      for (i,sb) in enumerate(sbdual)
        printstyled("Computing supremizers in space for dual field $i\n";color=:blue)
        cmat_i = assemble_constraint_matrix(feop,solver,snaps,i)
        supr_i = cmat_i*sb.bases_space[i]
        sbu_i = gram_schmidt(supr_i,sbu.array)
        sbu.array = hcat(sbu.array,sbu_i)
      end
      return
    end

    function assemble_constraint_matrix(
      feop::$Top,
      solver::$Tslv,
      snaps::MultiFieldSnapshots,
      i::Int)

      sols,params = get_data(snaps)
      filter = (1,i)
      assemble_matrix(feop,solver,sols,params,filter)
    end
  end
end

function add_time_supremizers!(bases_time::Vector{<:NnzArray};kwargs...)
  tbu,tbdual... = bases_time
  for (i,tb) in enumerate(tbdual)
    printstyled("Computing supremizers in time for dual field $i\n";color=:blue)
    tbu_i = add_time_supremizers([tbu.array,tb.array];kwargs...)
    tbu.array = tbu_i
  end
  return
end

function add_time_supremizers(bases_time::Vector{<:AbstractMatrix};ttol=1e-2)
  basis_u,basis_p = bases_time
  basis_up = basis_u'*basis_p

  function enrich(
    basis_u::AbstractMatrix{Float},
    basis_up::AbstractMatrix{Float},
    v::AbstractArray{Float})

    vnew = orth_complement(v,basis_u)
    vnew /= norm(vnew)
    hcat(basis_u,vnew),vcat(basis_up,vnew'*basis_p)
  end

  count = 0
  ntp_minus_ntu = size(basis_p,2) - size(basis_u,2)
  if ntp_minus_ntu > 0
    for ntp = 1:ntp_minus_ntu
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
    end
  end

  ntp = 1
  while ntp ≤ size(basis_up,2)
    proj = ntp == 1 ? zeros(size(basis_up[:,1])) : orth_projection(basis_up[:,ntp],basis_up[:,1:ntp-1])
    dist = norm(basis_up[:,1]-proj)
    if dist ≤ ttol
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
      ntp = 0
    else
      basis_up[:,ntp] -= proj
    end
    ntp += 1
  end

  printstyled("Added $count time supremizers\n";color=:blue)
  basis_u
end

function save(info::RBInfo,rb::RBSpace)
  if info.save_offline
    path = joinpath(info.offline_path,"basis")
    save(path,rb)
  end
end

function load(T::Type{RBSpace},info::RBInfo)
  path = joinpath(info.offline_path,"basis")
  load(T,path)
end

# REMOVE IN THE FUTURE
function save(info::RBInfo,rbspace::Union{RBSpace,TransientRBSpace})
  if info.save_offline
    path = joinpath(info.offline_path,"basis")
    save(path,rbspace)
  end
end

function load(T::Union{Type{RBSpace},Type{TransientRBSpace}},info::RBInfo)
  path = joinpath(info.offline_path,"basis")
  load(T,path)
end
