function rb(info::RBInfo,args...;tt=TimeTracker(OfflineTime(0.,0.),0.))
  snaps, = args
  if info.load_offline
    load_rb(info,snaps)
  else
    tt.offline_time.basis_time += @elapsed begin
      assemble_rb(info,args...)
    end
  end
end

function load_rb(info::RBInfo,snaps::Snapshots)
  id = get_id(snaps)
  load_rb(info,id)
end

function load_rb(info::RBInfo,snaps::NTuple{N,Snapshots}) where N
  Broadcasting(si -> load_rb(info,si))(snaps)
end

function assemble_rb(
  info::RBInfoSteady,
  snaps::Snapshots,
  args...)

  id = get_id(snaps)
  basis_space = rb_space(info,snaps)
  RBSpaceSteady(id,basis_space)
end

function assemble_rb(
  info::RBInfoUnsteady,
  snaps::Snapshots,
  args...)

  id = get_id(snaps)
  basis_space = rb_space(info,snaps)
  basis_time = rb_time(info,snaps,basis_space)
  RBSpaceUnsteady(id,basis_space,basis_time)
end

function assemble_rb(
  info::RBInfoSteady,
  snaps::NTuple{2,Snapshots},
  args...)

  def = isindef(info)
  snaps_u,snaps_p = snaps

  bs_u = rb_space(info,snaps_u)
  bs_p = rb_space(info,snaps_p)
  bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),args...)

  rbspace_u = RBSpaceSteady(get_id(snaps_u),bs_u_supr)
  rbspace_p = RBSpaceSteady(get_id(snaps_p),bs_p)

  rbspace_u,rbspace_p
end

function assemble_rb(
  info::RBInfoUnsteady,
  snaps::NTuple{2,Snapshots},
  args...)

  def = isindef(info)
  snaps_u,snaps_p = snaps
  opB,ph,μ,tol... = args

  bs_u = rb_space(info,snaps_u)
  bs_p = rb_space(info,snaps_p)
  bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),opB,ph,μ)
  bt_u = rb_time(info,snaps_u,bs_u)
  bt_p = rb_time(info,snaps_p,bs_p)
  bt_u_supr = add_time_supremizers(def,(bt_u,bt_p),tol...)

  rbspace_u = RBSpaceUnsteady(get_id(snaps_u),bs_u_supr,bt_u_supr)
  rbspace_p = RBSpaceUnsteady(get_id(snaps_p),bs_p,bt_p)

  rbspace_u,rbspace_p
end

function rb_space(
  info::RBInfo,
  snap::Snapshots)

  ϵ = info.ϵ
  printstyled("Spatial POD, tolerance: $ϵ\n";color=:blue)
  POD(snap;ϵ)
end

function rb_time(
  info::RBInfoUnsteady,
  snap::Snapshots,
  basis_space::Matrix{Float})

  ϵ = info.ϵ
  printstyled("Temporal POD, tolerance: $ϵ\n";color=:blue)

  s1 = get_snap(snap)
  ns = get_nsnap(snap)

  if info.time_red_method == "ST-HOSVD"
    s2 = mode2_unfolding(basis_space'*s1,ns)
  else
    s2 = mode2_unfolding(snap)
  end
  POD(s2;ϵ)
end

function add_space_supremizers(
  ::Val{false},
  basis::NTuple{2,Matrix{Float}},
  args...)

  first(basis)
end

function add_space_supremizers(
  ::Val{true},
  basis::NTuple{2,Matrix{Float}},
  opB::ParamBilinOperator,
  ph::Snapshots,
  μ::Vector{Param})

  basis_u, = basis
  supr = space_supremizers(basis,opB,ph,μ)
  hcat(basis_u,supr)
end

function space_supremizers(
  basis::NTuple{2,Matrix{Float}},
  opB::ParamBilinOperator,
  ph::Snapshots,
  μ::Vector{Param})

  printstyled("Computing primal supremizers\n";color=:blue)

  basis_u,basis_p = basis
  constraint_mat = assemble_constraint_matrix(opB,basis_p,ph,μ)
  gram_schmidt(constraint_mat,basis_u)
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator{Affine,Ttr},
  basis_p::Matrix{Float},
  ::Snapshots,
  μ::Vector{Param}) where Ttr

  @assert opB.id == :B
  printstyled("Fetching supremizing operator Bᵀ\n";color=:blue)

  B = assemble_affine_quantity(opB)
  Matrix(B')*basis_p
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator,
  ::Matrix{Float},
  ph::Snapshots,
  μ::Vector{Param})

  @assert opB.id == :B
  printstyled("Bᵀ is nonaffine: must assemble the constraint matrix\n";color=:blue)

  B = assemble_matrix(opB;μ,t=first(get_timesθ(opB)))
  Brb(k::Int) = Matrix(B[k]')*ph.snaps[:,k]
  Matrix(Brb.(axes(basis_p,2)))
end

function add_time_supremizers(
  ::Val{false},
  basis::NTuple{2,Matrix{Float}},
  args...)

  first(basis)
end

function add_time_supremizers(
  ::Val{true},
  basis::NTuple{2,Matrix{Float}},
  tol=1e-2)

  printstyled("Checking if supremizers in time need to be added\n";color=:blue)

  basis_u,basis_p = basis
  basis_up = basis_u'*basis_p

  function enrich(basis_u::Matrix{Float},basis_up::Matrix{Float},v::Vector)
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
    printstyled("Distance measure of basis vector number $ntp is: $dist\n";color=:blue)
    if dist ≤ 1e-2
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
