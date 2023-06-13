function reduce_fe_space(
  info::RBInfo,
  feop::ParamFEOperator,
  fe_solver::FESolver;
  kwargs...)

  n_snaps = info.nsnaps
  s = generate_snapshots(feop,fe_solver,n_snaps)
  save(info,s)
  RBSpace(s;kwargs...)
end

function reduce_fe_space(
  info::RBInfo,
  feop::ParamTransientFEOperator,
  fe_solver::ODESolver;
  kwargs...)

  n_snaps = info.nsnaps
  s = generate_snapshots(feop,fe_solver,n_snaps)
  save(info,s)
  TransientRBSpace(s;kwargs...)
end

abstract type RBSpace{T} end
abstract type TransientRBSpace{T} end

struct SingleFieldRBSpace{T} <: RBSpace{T}
  basis_space::NnzMatrix{T}
end

struct MultiFieldRBSpace{T} <: RBSpace{T}
  basis_space::Vector{NnzMatrix{T}}
end

function RBSpace(s::SingleFieldSnapshots{T};kwargs...) where T
  basis_space = tpod(s;kwargs...)
  SingleFieldRBSpace{T}(basis_space)
end

function RBSpace(s::MultiFieldSnapshots{T};kwargs...) where T
  bases_space = multi_tpod(s;kwargs...)
  MultiFieldRBSpace{T}(bases_space)
end

struct TransientSingleFieldRBSpace{T} <: TransientRBSpace{T}
  basis_space::NnzMatrix{T}
  basis_time::NnzMatrix{T}
end

struct TransientMultiFieldRBSpace{T} <: TransientRBSpace{T}
  basis_space::Vector{NnzMatrix{T}}
  basis_time::Vector{NnzMatrix{T}}
end

function TransientRBSpace(s::SingleFieldSnapshots{T};kwargs...) where T
  basis_space,basis_time = transient_tpod(s;kwargs...)
  TransientSingleFieldRBSpace{T}(basis_space,basis_time)
end

function TransientRBSpace(s::MultiFieldSnapshots{T};kwargs...) where T
  bases_space,bases_time = multi_transient_tpod(s;kwargs...)
  TransientMultiFieldRBSpace{T}(bases_space,bases_time)
end

function add_space_supremizers!(space_bases::Vector{<:NnzMatrix};kwargs...)
  sbu,sbdual... = space_bases
  for sb in sbdual
    sbu_i,sbd_i = add_space_supremizers([sbu.array,sb.array];kwargs...)
    sbu.array = sbu_i
    sb.array = sbd_i
  end
  return
end

function assemble_constraint_matrix(
  feop::ParamTransientFEOperator,
  i::Int)

  trial_dual_field = trial[i]
  test_field = test[1]
  cmat = assemble_matrix(op.jac,op.trial,op.test)
end

# function add_space_supremizers(
#   ::Val{true},
#   basis::NTuple{2,AbstractMatrix{Float}},
#   opB::ParamBilinOperator)

#   basis_u, = basis
#   supr = assemble_space_supremizers(basis,opB)
#   hcat(basis_u,supr)
# end

# function assemble_space_supremizers(
#   basis::NTuple{2,AbstractMatrix{Float}},
#   opB::ParamBilinOperator)

#   printstyled("Computing supremizers in space\n";color=:blue)
#   basis_u,basis_p = basis
#   constraint_mat = assemble_constraint_matrix(opB,basis_p)
#   gram_schmidt(constraint_mat,basis_u)
# end

# function assemble_constraint_matrix(
#   opB::ParamBilinOperator{Affine,Ttr},
#   basis_p::AbstractMatrix{Float}) where Ttr

#   @assert opB.id == :B
#   B = assemble_affine_quantity(opB)
#   B'*basis_p
# end

# function assemble_constraint_matrix(
#   ::ParamBilinOperator,
#   ::AbstractMatrix{Float},
#   ::Snapshots)

#   error("Implement this")
# end

function add_time_supremizers!(time_bases::Vector{<:NnzMatrix};kwargs...)
  tbu,tbdual... = time_bases
  for tb in tbdual
    tbu_i,tbd_i = add_time_supremizers([tbu.array,tb.array];kwargs...)
    tbu.array = tbu_i
    tb.array = tbd_i
  end
  return
end

function add_time_supremizers(time_bases::Vector{<:AbstractMatrix};ttol=1e-2)
  printstyled("Checking if supremizers in time need to be added\n";color=:blue)

  basis_u,basis_p = time_bases
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
