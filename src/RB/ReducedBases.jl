get_rb(info::RBInfo) = get_rb(info,isindef(info))
get_rb(info::RBInfo,::Val{false}) = load_rb(info,:u)
get_rb(info::RBInfo,::Val{true}) = load_rb(info,:u),load_rb(info,:p)

function assemble_rb(info::RBInfoSteady,res::RBResults,snaps)
  basis_space = rb_space(info,res,snaps)
  RBSpaceSteady.(snaps,basis_space)
end

function assemble_rb(info::RBInfoUnsteady,res::RBResults,snaps)
  basis_space = rb_space(info,res,snaps)
  basis_time = rb_time(info,res,snaps,basis_space,isindef(info))
  RBSpaceUnsteady.(snaps,basis_space,basis_time)
end

function rb_space(
  info::RBInfo,
  res::RBResults,
  snaps,
  opB::ParamBilinOperator,
  μ::Snapshots)

  println("Spatial POD, tolerance: $(info.ϵ)")
  res.offline_time += @elapsed begin
    basis_space = POD(snaps,info.ϵ)
  end

  add_space_supremizers(isindef(info),basis_space,snaps,opB,μ)
end

function rb_time(
  info::RBInfoUnsteady,
  res::RBResults,
  snap::Snapshots,
  basis_space::Matrix,
  ::Val{true})

  println("Temporal POD, tolerance: $(info.ϵ)")

  s1 = get_snap(snap)
  Nt = get_Nt(snap)

  res.offline_time += @elapsed begin
    if info.time_red_method == "ST-HOSVD"
      s2 = mode2_unfolding(basis_space'*s1,Nt)
    else
      s2 = mode2_unfolding(s1,Nt)
    end
    basis_time = POD(s2,info.ϵ)
  end
  basis_time
end

function rb_time(
  info::RBInfoUnsteady,
  res::RBResults,
  snap::Vector{Snapshots},
  basis_space::Vector{Matrix},
  ::Val{true})

  println("Temporal POD, tolerance: $(info.ϵ)")

  s1 = get_snap.(snap)
  Nt = get_Nt.(snap)

  res.offline_time += @elapsed begin
    if info.time_red_method == "ST-HOSVD"
      s2 = mode2_unfolding.(basis_space'.*s1,Nt)
    else
      s2 = mode2_unfolding.(s1,Nt)
    end
    basis_time = Broadcating(s -> POD(s,info.ϵ))(s2)
  end
  add_time_supremizers(basis_time...)
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator{Affine,TT},
  basis_p::Matrix,
  ::Snapshots,
  μ::Snapshots) where TT

  @assert opB.id == :B
  println("Loading matrix Bᵀ")

  B = get_structure(opB)(first(μ))
  Brb(k::Int) = B*basis_p[:,k]
  Brb.(1:get_ns(bp))
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator,
  ::Matrix,
  ph::Snapshots,
  μ::Snapshots)

  @assert opB.id == :B
  println("Matrix Bᵀ is nonaffine: must assemble the constraint matrix")

  B = get_structure(opB).(μ)
  Brb(k::Int) = B[k]*ph.snaps[:,k]
  Brb.(1:get_ns(bp))
end

function space_supremizers(
  opB::ParamBilinOperator,
  basis_u::Matrix,
  basis_p::Matrix,
  ph::Snapshots,
  μ::Snapshots)

  println("Computing primal supremizers")

  constraint_mat = assemble_constraint_matrix(opB,basis_p,ph,μ)
  gram_schmidt!(constraint_mat,get_basis_space(basis_u))
end

add_space_supremizers(::Val{false},basis_u::Matrix,args...) = basis_u

function add_space_supremizers(
  ::Val{true},
  basis_u::Matrix,
  opB::ParamBilinOperator,
  ph::Snapshots,
  μ::Snapshots)

  supr = space_supremizers(opB,basis_u,basis_p,ph,μ)
  hcat(basis_u,supr)
end

function add_space_supremizers(
  ::Val{true},
  basis_u::Matrix,
  basis_p::Matrix,
  opB::ParamBilinOperator,
  ph::Snapshots,
  μ::Snapshots)

  supr = space_supremizers(opB,basis_u,basis_p,ph,μ)
  hcat(basis_u,supr)
end

function add_time_supremizers(
  basis_u::Matrix,
  basis_p::Matrix,
  tol=1e-2)

  println("Checking if supremizers in time need to be added")

  basis_up = basis_u'*basis_p
  count = 0

  function enrich(basis_u::Matrix{T},basis_up::Matrix{T},v::Vector)
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
  return
end
