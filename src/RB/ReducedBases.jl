get_rb(info::RBInfo) = get_rb(info,isindef(info))
get_rb(info::RBInfo,::Val{false}) = load_rb(info,:u)
get_rb(info::RBInfo,::Val{true}) = load_rb(info,:u),load_rb(info,:p)

function assemble_rb(info::RBInfoSteady,tt::TimeTracker,snaps,args...)
  tt.offline_time += @elapsed begin
    basis_space = rb_space(info,snaps,args...)
  end
  id = get_id(snaps)
  RBSpaceSteady(id,basis_space)
end

function assemble_rb(info::RBInfoUnsteady,tt::TimeTracker,snaps,args...)
  tt.offline_time += @elapsed begin
    basis_space = rb_space(info,snaps,args...)
    basis_time = rb_time(info,snaps,basis_space)
  end
  RBSpaceUnsteady(id,basis_space,basis_time)
end

function rb_space(
  info::RBInfo,
  snap::Snapshots)

  println("Spatial POD, tolerance: $(info.ϵ)")
  POD(snap,info.ϵ)
end

rb_space(info::RBInfo,snaps::Vector{Snapshots}) =
  Broadcasting(s->rb_space(info,s))(snaps)

function rb_time(
  info::RBInfoUnsteady,
  snap::Snapshots,
  basis_space::Matrix)

  println("Temporal POD, tolerance: $(info.ϵ)")

  s1 = get_snap(snap)
  Nt = get_Nt(snap)

  if info.time_red_method == "ST-HOSVD"
    s2 = mode2_unfolding(basis_space'.*s1,Nt)
  else
    s2 = mode2_unfolding(s1,Nt)
  end
  POD(s2,info.ϵ)
end

rb_time(info::RBInfo,tt::TimeTracker,snaps::Vector{Snapshots},basis_space::Vector{Matrix}) =
  Broadcasting((s,b)->rb_time(info,tt,s,b))(snaps,basis_space)

function add_space_supremizers(
  opB::ParamBilinOperator,
  basis::Vector{Matrix},
  ph::Snapshots,
  μ::Vector{Param})

  basis_u,basis_p = basis
  supr = space_supremizers(opB,basis_u,basis_p,ph,μ)
  hcat(basis_u,supr)
end

function space_supremizers(
  opB::ParamBilinOperator,
  basis_u::Matrix,
  basis_p::Matrix,
  ph::Snapshots,
  μ::Vector{Param})

  println("Computing primal supremizers")

  constraint_mat = assemble_constraint_matrix(opB,basis_p,ph,μ)
  gram_schmidt!(constraint_mat,get_basis_space(basis_u))
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator{Affine,TT},
  basis_p::Matrix,
  ::Snapshots,
  μ::Vector{Param}) where TT

  @assert opB.id == :B
  println("Loading matrix Bᵀ")

  B = assemble_matrix(opB)(first(μ))
  Brb(k::Int) = B*basis_p[:,k]
  Brb.(1:get_ns(bp))
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator,
  ::Matrix,
  ph::Snapshots,
  μ::Vector{Param})

  @assert opB.id == :B
  println("Matrix Bᵀ is nonaffine: must assemble the constraint matrix")

  B = assemble_matrix(opB).(μ)
  Brb(k::Int) = B[k]*ph.snaps[:,k]
  Brb.(1:get_ns(bp))
end

function add_time_supremizers(
  basis::Vector{Matrix},
  tol=1e-2)

  println("Checking if supremizers in time need to be added")

  basis_u,basis_p = basis
  basis_up = basis_u'*basis_p
  count = 0

  function enrich(basis_u::Matrix,basis_up::Matrix,v::Vector)
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
