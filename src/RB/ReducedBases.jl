function rb(
  info::RBInfo,
  snaps::NTuple{N,Snapshots},
  args...;
  kwargs...)::NTuple{N,RBSpace} where N

  if info.load_offline
    printstyled("Loading reduced bases\n";color=:red)
    load(info,get_id(snaps))
  else
    printstyled("Assembling reduced bases\n";color=:red)
    assemble_rb(info,snaps,args...;kwargs...)
  end
end

function assemble_rb(
  info::RBInfoSteady,
  snaps::Snapshots;
  kwargs...)

  id = get_id(snaps)
  basis_space = rb_space(snaps;ϵ=info.ϵ,kwargs...)
  (RBSpaceSteady(id,basis_space),)
end

function assemble_rb(
  info::RBInfoUnsteady,
  snaps::Snapshots;
  kwargs...)

  id = get_id(snaps)
  basis_space = rb_space(snaps;ϵ=info.ϵ,kwargs...)
  basis_time = rb_time(snaps,basis_space;ϵ=info.ϵ,kwargs...)
  (RBSpaceUnsteady(id,basis_space,basis_time),)
end

function assemble_rb(
  info::RBInfoSteady,
  snaps::NTuple{2,Snapshots},
  args...;
  kwargs...)

  def = isindef(info)
  snaps_u,snaps_p = snaps

  bs_u = rb_space(snaps_u;ϵ=info.ϵ,kwargs...)
  bs_p = rb_space(snaps_p;ϵ=info.ϵ,kwargs...)
  bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),args...)

  rbspace_u = RBSpaceSteady(get_id(snaps_u),bs_u_supr)
  rbspace_p = RBSpaceSteady(get_id(snaps_p),bs_p)

  (rbspace_u,rbspace_p)
end

function assemble_rb(
  info::RBInfoUnsteady,
  snaps::NTuple{2,Snapshots},
  args...;
  kwargs...)

  def = isindef(info)
  snaps_u,snaps_p = snaps
  opB,ttol... = args

  bs_u = rb_space(snaps_u;ϵ=info.ϵ,kwargs...)
  bs_p = rb_space(snaps_p;ϵ=info.ϵ,kwargs...)
  bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),opB)
  bt_u = rb_time(snaps_u,bs_u;ϵ=info.ϵ,kwargs...)
  bt_p = rb_time(snaps_p,bs_p;ϵ=info.ϵ,kwargs...)
  bt_u_supr = add_time_supremizers(def,(bt_u,bt_p),ttol...)

  rbspace_u = RBSpaceUnsteady(get_id(snaps_u),bs_u_supr,bt_u_supr)
  rbspace_p = RBSpaceUnsteady(get_id(snaps_p),bs_p,bt_p)

  (rbspace_u,rbspace_p)
end

function rb_space(snap::Snapshots;kwargs...)
  POD(snap;kwargs...)
end

function rb_time(
  snap::Snapshots,
  basis_space::AbstractMatrix{Float};
  kwargs...)

  s1 = get_snap(snap)
  ns = get_nsnap(snap)
  s2 = mode2_unfolding(basis_space'*s1,ns)
  POD(s2;kwargs...)
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
  supr = space_supremizers(basis,opB)
  hcat(basis_u,supr)
end

function space_supremizers(
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
