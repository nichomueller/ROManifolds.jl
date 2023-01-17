function rb(info::RBInfo,args...)
  _,snaps,_ = args
  def = isindef(info)
  info.load_offline ? load_rb(info,snaps) : assemble_rb(def,info,args...)
end

function load_rb(info::RBInfo,snaps::Snapshots)
  id = get_id(snaps)
  load_rb(info,id)
end

function load_rb(info::RBInfo,snaps::NTuple{N,Snapshots}) where N
  Broadcasting(si -> load_rb(info,si))(snaps)
end

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
    bs_u = rb_space(info,snaps_u)
    bs_p = rb_space(info,snaps_p)
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
  opB,ph,μ,tol... = args

  tt.offline_time += @elapsed begin
    bs_u = rb_space(info,snaps_u)
    bs_p = rb_space(info,snaps_p)
    bs_u_supr = add_space_supremizers((bs_u,bs_p),opB,ph,μ)
    bt_u = rb_time(info,snaps_u,bs_u)
    bt_p = rb_time(info,snaps_p,bs_p)
    bt_u_supr = add_time_supremizers((bt_u,bt_p),tol...)
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

matrix_B(opB::ParamSteadyBilinOperator{Affine,Ttr},μ::Vector{Param}) where Ttr =
  assemble_matrix(opB)(first(μ))
matrix_B(opB::ParamSteadyBilinOperator,μ::Vector{Param}) =
  assemble_matrix(opB).(μ)
matrix_B(opB::ParamUnsteadyBilinOperator{Affine,Ttr},μ::Vector{Param}) where Ttr =
  assemble_matrix(opB,realization(opB.tinfo))(first(μ))
matrix_B(opB::ParamUnsteadyBilinOperator,μ::Vector{Param}) =
  assemble_matrix(opB,realization(opB.tinfo)).(μ)

function assemble_constraint_matrix(
  opB::ParamBilinOperator{Affine,Ttr},
  basis_p::Matrix{Float},
  ::Snapshots,
  μ::Vector{Param}) where Ttr

  @assert opB.id == :B
  println("Fetching Bᵀ")

  B = matrix_B(opB,μ)
  Matrix(B')*basis_p
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator,
  ::Matrix{Float},
  ph::Snapshots,
  μ::Vector{Param})

  @assert opB.id == :B
  println("Bᵀ is nonaffine: must assemble the constraint matrix")

  B = matrix_B(opB,μ)
  Brb(k::Int) = Matrix(B[k]')*ph.snaps[:,k]
  Matrix(Brb.(axes(basis_p,2)))
end

function add_time_supremizers(
  basis::NTuple{2,Matrix{Float}},
  tol=1e-2)

  println("Checking if supremizers in time need to be added")

  basis_u,basis_p = basis
  basis_up = basis_u'*basis_p
  count = 0

  function enrich(basis_u::Matrix{Float},basis_up::Matrix{Float},v::Vector)
    vnew = orth_complement(v,basis_u)
    vnew /= norm(vnew)
    hcat(basis_u,vnew),vcat(basis_up,vnew'*basis_p)
  end

  for ntp = axes(basis_up,2)
    dist = ntp == 1 ? norm(basis_up[:,1]) : norm(orth_projection(basis_up[:,ntp],basis_up[:,1:ntp-1]))
    println("Distance measure of basis vector number $ntp is: $dist")
    if dist ≤ tol
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
    end
  end

  println("Added $count time supremizers")
  basis_u
end
