function rb(info::RBInfo,args...)
  info.load_offline ? get_rb(isindef(info),info) : assemble_rb(isindef(info),info,args...)
end

get_rb(::Val{false},info::RBInfo) = load_rb(info,:u)
get_rb(::Val{true},info::RBInfo) = load_rb(info,:u),load_rb(info,:p)

function assemble_rb(
  ::Val{false},
  info::RBInfoSteady,
  tt::TimeTracker,
  snaps::Snapshots,
  args...)

  id = get_id(snaps)
  tt.offline_time += @elapsed begin
    basis_space = rb_space(info,snaps)
  end

  rbspace = RBSpaceSteady(id,basis_space)
  save(info,rbspace)

  rbspace
end

function assemble_rb(
  ::Val{false},
  info::RBInfoUnsteady,
  tt::TimeTracker,
  snaps::Snapshots,
  args...)

  id = get_id(snaps)
  tt.offline_time += @elapsed begin
    basis_space = rb_space(info,snaps)
    basis_time = rb_time(info,snaps,basis_space)
  end

  rbspace = RBSpaceUnsteady(id,basis_space,basis_time)
  save(info,rbspace)

  rbspace
end

function assemble_rb(
  ::Val{true},
  info::RBInfoSteady,
  tt::TimeTracker,
  snaps::NTuple{2,Snapshots},
  args...)

  snaps_u,snaps_p = snaps

  tt.offline_time += @elapsed begin
    bs_p = rb_space(info,snaps_p)
    bs_u = rb_space(info,snaps_u)
    bs_u_supr = add_space_supremizers((bs_u,bs_p),args...)
  end

  rbspace_u = RBSpaceSteady(get_id(snaps_u),bs_u_supr)
  rbspace_p = RBSpaceSteady(get_id(snaps_p),bs_p)
  save(info,rbspace_u)
  save(info,rbspace_p)

  rbspace_u,rbspace_p
end

function assemble_rb(
  ::Val{true},
  info::RBInfoUnsteady,
  tt::TimeTracker,
  snaps::NTuple{2,Snapshots},
  args...)

  snaps_u,snaps_p = snaps

  tt.offline_time += @elapsed begin
    bs_p = rb_space(info,snaps_p)
    bs_u = rb_space(info,snaps_u)
    bs_u_supr = add_space_supremizers((bs_u,bs_p),args...)
    bt_p = rb_time(info,snaps_p,bs_p)
    bt_u = rb_time(info,snaps_u,bs_u)
    bt_u_supr =add_time_supremizers((bt_u,bt_p))
  end

  rbspace_u = RBSpaceUnsteady(get_id(snaps_u),bs_u_supr,bt_u_supr)
  rbspace_p = RBSpaceUnsteady(get_id(snaps_p),bs_p,bt_p)
  save(info,rbspace_u)
  save(info,rbspace_p)

  rbspace_u,rbspace_p
end

function rb_space(
  info::RBInfo,
  snap::Snapshots)

  println("Spatial POD, tolerance: $(info.ϵ)")
  POD(snap;ϵ=info.ϵ)
end

function rb_time(
  info::RBInfoUnsteady,
  snap::Snapshots,
  basis_space::Matrix{Float})

  println("Temporal POD, tolerance: $(info.ϵ)")

  s1 = get_snap(snap)
  ns = get_nsnap(snap)

  if info.time_red_method == "ST-HOSVD"
    s2 = mode2_unfolding(basis_space'*s1,ns)
  else
    s2 = mode2_unfolding(snap)
  end
  POD(s2;ϵ=info.ϵ)
end

function add_space_supremizers(
  basis::NTuple{N,Matrix{Float}},
  opB::ParamBilinOperator,
  ph::Snapshots,
  μ::Vector{Param}) where N

  basis_u, = basis
  supr = space_supremizers(basis,opB,ph,μ)
  hcat(basis_u,supr)
end

function space_supremizers(
  basis::NTuple{N,Matrix{Float}},
  opB::ParamBilinOperator,
  ph::Snapshots,
  μ::Vector{Param}) where N

  println("Computing primal supremizers")

  basis_u,basis_p = basis
  constraint_mat = assemble_constraint_matrix(opB,basis_p,ph,μ)
  gram_schmidt(constraint_mat,basis_u)
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator{Affine,TT},
  basis_p::Matrix{Float},
  ::Snapshots,
  μ::Vector{Param}) where TT

  @assert opB.id == :B
  println("Fetching Bᵀ")

  B = assemble_matrix(opB)(first(μ))
  Matrix(B')*basis_p
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator,
  ::Matrix{Float},
  ph::Snapshots,
  μ::Vector{Param})

  @assert opB.id == :B
  println("Bᵀ is nonaffine: must assemble the constraint matrix")

  B = assemble_matrix(opB)(μ)
  Brb(k::Int) = Matrix(B[k]')*ph.snaps[:,k]
  Matrix(Brb.(axes(basis_p,2)))
end

function add_time_supremizers(
  basis::NTuple{N,Matrix{Float}},
  tol=1e-2) where N

  println("Checking if supremizers in time need to be added")

  basis_u,basis_p = basis
  basis_up = basis_u'*basis_p
  count = 0

  function enrich(basis_u::Matrix{Float},basis_up::Matrix{Float},v::Vector)
    vnew = orth_complement(v,basis_up)
    vnew /= norm(vnew)
    hcat(basis_u,vnew),hcat(basis_up,vnew'*bp)
  end

  dist = norm(bp[:,1])
  for np = 1:get_nt(bp)
    println("Distance measure of basis vector number $np is: $dist")
    if dist ≤ tol
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,np])
      count += 1
    end
    dist = orth_projection(basis_up[:,np],basis_up[:,1:np-1])
  end

  println("Added $count time supremizers")
  basis_u,basis_p
end
