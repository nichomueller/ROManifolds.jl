abstract type RBFunction <: FEFunction end

abstract type RBBasis end

struct SingleFieldRBBasis <: RBBasis
  basis_space::AbstractArray

  function SingleFieldRBBasis(snaps::Snapshots;kwargs...)
    basis_space = POD(snaps;kwargs...)
    new(basis_space)
  end
end

get_basis_space(rbb::SingleFieldRBBasis) = rbb.basis_space

function similar_rb_basis(::SingleFieldRBBasis,basis_space)
  SingleFieldRBBasis(basis_space)
end

function reduce_fe_space(feop;load_offline=true,kwargs...)
  reduce_fe_space(Val{load_offline}(),feop;args...)
end

function reduce_fe_space(::Val{true},feop;kwargs...)
  load_rb_space()
end

function reduce_fe_space(::Val{false},feop;n_snaps=50,fe_solver,kwargs...)
  snaps = generate_fe_snapshots(feop,n_snaps,fe_solver)
  Ns = get_Ns(feop)
  assemble_rb_space(snaps,Ns;kwargs...)
end

function assemble_rb_space(snaps::Snapshots,::Int;kwargs...)
  SingleFieldRBBasis(snaps;kwargs...)
end

struct TransientSingleFieldRBBasis <: RBBasis
  basis_space::AbstractArray
  basis_time::AbstractArray

  function TransientSingleFieldRBBasis(snaps::Snapshots;kwargs...)
    basis_space,basis_time = assemble_spatio_temporal_rb(snaps;kwargs...)
    new(basis_space,basis_time)
  end
end

function assemble_spatio_temporal_rb(snaps::Snapshots;kwargs...)
  nrows,ncols = size(get_snap(snaps))
  assemble_spatio_temporal_rb(Val{nrows > ncols}(),snap;kwargs...)
end

function assemble_spatio_temporal_rb(::Val{false},snaps::Snapshots;kwargs...)
  snaps_space = get_snap(snaps)
  nsnaps = length(snaps)
  basis_space = POD(snaps_space;kwargs...)
  rb_snaps_time = mode2_unfolding(basis_space'*snaps_space,nsnaps)
  basis_time = POD(rb_snaps_time;kwargs...)

  basis_space,basis_time
end

function assemble_spatio_temporal_rb(::Val{true},snaps::Snapshots;kwargs...)
  snaps_space = get_snap(snaps)
  nsnaps = length(snaps)
  snaps_time = mode2_unfolding(snaps_space,nsnaps)
  basis_time = POD(snaps_time;kwargs...)
  red_snaps_space = mode2_unfolding(basis_time'*snaps_time,nsnaps)
  basis_space = POD(red_snaps_space;kwargs...)

  basis_space,basis_time
end

struct MultiFieldRBBasisComponent <: RBBasis
  basis_space::AbstractArray

  function SingleFieldRBBasis(snaps::Snapshots;kwargs...)
    basis_space = POD(snaps;kwargs...)
    new(basis_space)
  end
end

function assemble_rb_space(
  info::RBInfoSteady,
  snaps_u::Snapshots,
  snaps_p::Snapshots,
  args...;
  kwargs...)

  def = isindef(info)

  bs_u = assemble_spatial_rb(snaps_u;ϵ=info.ϵ,kwargs...)
  bs_p = assemble_spatial_rb(snaps_p;ϵ=info.ϵ,kwargs...)
  bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),args...)

  rbspace_u = RBSpaceSteady(get_id(snaps_u),bs_u_supr)
  rbspace_p = RBSpaceSteady(get_id(snaps_p),bs_p)

  (rbspace_u,rbspace_p)
end

function assemble_rb_space(
  info::RBInfoUnsteady,
  snaps_u::Snapshots,
  snaps_p::Snapshots,
  args...;
  kwargs...)

  def = isindef(info)
  opB,ttol... = args

  bs_u,bt_u = assemble_spatio_temporal_rb(snaps_u;ϵ=info.ϵ,kwargs...)
  bs_p,bt_p = assemble_spatio_temporal_rb(snaps_p;ϵ=info.ϵ,kwargs...)
  bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),opB)
  bt_u_supr = add_time_supremizers(def,(bt_u,bt_p),ttol...)

  rbspace_u = RBSpaceUnsteady(get_id(snaps_u),bs_u_supr,bt_u_supr)
  rbspace_p = RBSpaceUnsteady(get_id(snaps_p),bs_p,bt_p)

  (rbspace_u,rbspace_p)
end

function add_space_supremizers(
  ::Val{false},
  basis::NTuple{2,AbstractMatrix{Float}},
  args...)

  first(basis)
end

function add_space_supremizers(
  ::Val{true},
  basis::NTuple{2,AbstractMatrix{Float}},
  opB::ParamBilinOperator)

  basis_u, = basis
  supr = assemble_space_supremizers(basis,opB)
  hcat(basis_u,supr)
end

function assemble_space_supremizers(
  basis::NTuple{2,AbstractMatrix{Float}},
  opB::ParamBilinOperator)

  printstyled("Computing supremizers in space\n";color=:blue)
  basis_u,basis_p = basis
  constraint_mat = assemble_constraint_matrix(opB,basis_p)
  gram_schmidt(constraint_mat,basis_u)
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator{Affine,Ttr},
  basis_p::AbstractMatrix{Float}) where Ttr

  @assert opB.id == :B
  B = assemble_affine_quantity(opB)
  B'*basis_p
end

function assemble_constraint_matrix(
  ::ParamBilinOperator,
  ::AbstractMatrix{Float},
  ::Snapshots)

  error("Implement this")
end

function add_time_supremizers(
  ::Val{false},
  basis::NTuple{2,AbstractMatrix{Float}},
  args...)

  first(basis)
end

function add_time_supremizers(
  ::Val{true},
  basis::NTuple{2,AbstractMatrix{Float}},
  ttol=1e-2)

  printstyled("Checking if supremizers in time need to be added\n";color=:blue)

  basis_u,basis_p = basis
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
