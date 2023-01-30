function rb(info::RBInfo,args...;kwargs...)
  _,snaps, = args
  info.load_offline ? load_rb(info,snaps) : assemble_rb(info,args...;kwargs...)
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
  tt::TimeTracker,
  snaps::Snapshots,
  args...;
  kwargs...)

  id = get_id(snaps)
  tt.offline_time.basis_time += @elapsed begin
    basis_space = rb_space(info,snaps;kwargs...)
  end

  rbspace = RBSpaceSteady(id,basis_space)
  save(info,rbspace)

  rbspace
end

function assemble_rb(
  info::RBInfoUnsteady,
  tt::TimeTracker,
  snaps::Snapshots,
  args...;
  kwargs...)

  id = get_id(snaps)
  tt.offline_time.basis_time += @elapsed begin
    basis_space = rb_space(info,snaps;kwargs...)
    basis_time = rb_time(info,snaps,basis_space)
  end

  rbspace = RBSpaceUnsteady(id,basis_space,basis_time)
  save(info,rbspace)

  rbspace
end

function assemble_rb(
  info::RBInfoSteady,
  tt::TimeTracker,
  snaps::NTuple{2,Snapshots},
  args...;
  kwargs...)

  def = isindef(info)
  snaps_u,snaps_p = snaps

  tt.offline_time.basis_time += @elapsed begin
    bs_u = rb_space(info,snaps_u;kwargs...)
    bs_p = rb_space(info,snaps_p;kwargs...)
    bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),args...)
  end

  rbspace_u = RBSpaceSteady(get_id(snaps_u),bs_u_supr)
  rbspace_p = RBSpaceSteady(get_id(snaps_p),bs_p)
  save(info,rbspace_u)
  save(info,rbspace_p)

  rbspace_u,rbspace_p
end

function assemble_rb(
  info::RBInfoUnsteady,
  tt::TimeTracker,
  snaps::NTuple{2,Snapshots},
  args...;
  kwargs...)

  def = isindef(info)
  snaps_u,snaps_p = snaps
  opB,ph,μ,tol... = args

  tt.offline_time.basis_time += @elapsed begin
    bs_u = rb_space(info,snaps_u;kwargs...)
    bs_p = rb_space(info,snaps_p)
    bs_u_supr = add_space_supremizers(def,(bs_u,bs_p),opB,ph,μ)
    bt_u = rb_time(info,snaps_u,bs_u;kwargs...)
    bt_p = rb_time(info,snaps_p,bs_p)
    bt_u_supr = add_time_supremizers(def,(bt_u,bt_p),tol...)
  end

  rbspace_u = RBSpaceUnsteady(get_id(snaps_u),bs_u_supr,bt_u_supr)
  rbspace_p = RBSpaceUnsteady(get_id(snaps_p),bs_p,bt_p)
  save(info,rbspace_u)
  save(info,rbspace_p)

  rbspace_u,rbspace_p
end

function rb_space(
  info::RBInfo,
  snap::Snapshots;
  ϵ=info.ϵ,sparsity=false)

  printstyled("\n Spatial POD, tolerance: $ϵ";color=:blue)
  if sparsity
    i,j,snap_nnz = findnz(snap)
    basis_space_nnz = POD(snap_nnz;ϵ=ϵ)
    Ns,ns = size(get_snap(snap),1),size(basis_space_nnz,2)
    nz = length(basis_space_nnz[:])
    basis_space = Matrix(sparse(i[1:nz],j[1:nz],basis_space_nnz[:],Ns,ns))
  else
    basis_space = POD(snap;ϵ=ϵ)
  end
  basis_space
end

function rb_time(
  info::RBInfoUnsteady,
  snap::Snapshots,
  basis_space::Matrix{Float};
  ϵ=info.ϵ)

  printstyled("\n Temporal POD, tolerance: $ϵ";color=:blue)

  s1 = get_snap(snap)
  ns = get_nsnap(snap)

  if info.time_red_method == "ST-HOSVD"
    s2 = mode2_unfolding(basis_space'*s1,ns)
  else
    s2 = mode2_unfolding(snap)
  end
  POD(s2;ϵ=ϵ)
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

  printstyled("\n Computing primal supremizers";color=:blue)

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
  printstyled("\n Fetching supremizing operator Bᵀ";color=:blue)

  B = matrix_B(opB,μ)
  Matrix(B')*basis_p
end

function assemble_constraint_matrix(
  opB::ParamBilinOperator,
  ::Matrix{Float},
  ph::Snapshots,
  μ::Vector{Param})

  @assert opB.id == :B
  printstyled("\n Bᵀ is nonaffine: must assemble the constraint matrix";color=:blue)

  B = matrix_B(opB,μ)
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

  printstyled("\n Checking if supremizers in time need to be added";color=:blue)

  basis_u,basis_p = basis
  basis_up = basis_u'*basis_p

  function enrich(basis_u::Matrix{Float},basis_up::Matrix{Float},v::Vector)
    vnew = orth_complement(v,basis_u)
    vnew /= norm(vnew)
    hcat(basis_u,vnew),vcat(basis_up,vnew'*basis_p)
  end

  count = 0
  ntp = 1
  while ntp ≤ size(basis_up,2)
    proj = ntp == 1 ? zeros(size(basis_up[:,1])) : orth_projection(basis_up[:,ntp],basis_up[:,1:ntp-1])
    dist = norm(basis_up[:,1]-proj)
    printstyled("\n Distance measure of basis vector number $ntp is: $dist";color=:blue)
    if dist ≤ 1e-2
      basis_u,basis_up = enrich(basis_u,basis_up,basis_p[:,ntp])
      count += 1
      ntp = 0
    else
      basis_up[:,ntp] -= proj
    end
    ntp += 1
  end

  printstyled("\n Added $count time supremizers";color=:blue)
  basis_u
end
