include("MDEIMBases.jl")

abstract type MDEIM end

struct MDEIMSteady <: MDEIM
  rb_space::RBSpaceSteady
  rb_lu::LU
  idx::Vector{Int}
  red_measure::Measure
end

struct MDEIMUnsteady <: MDEIM
  rb_space::RBSpaceUnsteady
  rb_lu::LU
  idx::NTuple{2,Vector{Int}}
  red_measure::Measure
end

function MDEIM(
  red_rbspace::RBSpaceSteady,
  rb_lu::LU,
  idx::Vector{Int},
  red_measure::Measure)

  MDEIMSteady(red_rbspace,rb_lu,idx,red_measure)
end

function MDEIM(
  red_rbspace::RBSpaceUnsteady,
  rb_lu::LU,
  idx::NTuple{2,Vector{Int}},
  red_measure::Measure)

  MDEIMUnsteady(red_rbspace,rb_lu,idx,red_measure)
end

function MDEIM(
  info::RBInfo,
  op::RBVariable,
  μ::Vector{Param},
  meas::Measure,
  args...)

  μ_mdeim = μ[1:info.mdeim_nsnap]
  rbspace,findnz_idx = mdeim_basis(info,op,μ_mdeim,args...)
  red_rbspace = project_mdeim_basis(op,rbspace,findnz_idx)
  idx = mdeim_idx(rbspace)
  rb_lu = get_rb_lu(info,rbspace,idx)
  idx = recast_in_full_dim(idx,findnz_idx)
  red_meas = get_red_measure(op,idx,meas)

  MDEIM(red_rbspace,rb_lu,idx,red_meas)
end

get_rbspace(mdeim::MDEIM) = mdeim.rb_space

get_rb_lu(mdeim::MDEIM) = mdeim.rb_lu

get_id(mdeim::MDEIM) = get_id(get_rbspace(mdeim))

get_basis_space(mdeim::MDEIM) = get_basis_space(get_rbspace(mdeim))

get_basis_time(mdeim::MDEIMUnsteady) = get_basis_time(get_rbspace(mdeim))

get_idx_space(mdeim::MDEIMSteady) = mdeim.idx

get_idx_space(mdeim::MDEIMUnsteady) = first(mdeim.idx)

get_idx_time(mdeim::MDEIMUnsteady) = last(mdeim.idx)

get_red_measure(mdeim::MDEIM) = mdeim.red_measure

function project_mdeim_basis(
  op::RBSteadyVariable,
  rb_space::RBSpace,
  args...)

  id = get_id(rb_space)
  bs = rb_space_projection(op,rb_space,args...)
  RBSpaceSteady(id,bs)
end

function project_mdeim_basis(
  op::RBUnsteadyVariable,
  rb_space::RBSpace,
  args...)

  id = get_id(rb_space)
  bs = rb_space_projection(op,rb_space,args...)
  bt = get_basis_time(rb_space)
  RBSpaceUnsteady(id,bs,bt)
end

function rb_space_projection(
  op::RBVariable,
  rb_space::RBSpace,
  args...)

  rb_space_projection(op,get_basis_space(rb_space),args...)
end

function rb_space_projection(
  op::RBLinVariable,
  basis_space::AbstractMatrix{Float},
  findnz_idx::Vector{Int})

  rbspace_row = get_rbspace_row(op)
  Ns = get_Ns(rbspace_row)
  full_basis_space = zeros(Ns,size(basis_space,2))
  full_basis_space[findnz_idx,:] = basis_space

  brow = get_basis_space(rbspace_row)
  brow'*full_basis_space
end

function rb_space_projection(
  op::RBBilinVariable,
  basis_space::AbstractMatrix{Float},
  findnz_idx::Vector{Int})

  rbspace_row = get_rbspace_row(op)
  brow = get_basis_space(rbspace_row)
  rbspace_col = get_rbspace_col(op)
  bcol = get_basis_space(rbspace_col)

  Qs = size(basis_space,2)
  Ns = get_Ns(rbspace_col)
  isparse,jsparse = from_vec_to_mat_idx(findnz_idx,Ns)
  red_basis_space = allocate_matrix(Matrix{Float},
    get_ns(rbspace_row)*get_ns(rbspace_col),Qs)
  for q = 1:Qs
    vsparse = Vector(basis_space[:,q])
    smat = sparse(isparse,jsparse,vsparse,Ns,Ns)
    red_basis_space[:,q] = Vector(brow'*smat*bcol)
  end

  red_basis_space
end

function mdeim_idx(rb_space::RBSpaceSteady)
  idx_space = mdeim_idx(get_basis_space(rb_space))
  idx_space
end

function mdeim_idx(rb_space::RBSpaceUnsteady)
  idx_space = mdeim_idx(get_basis_space(rb_space))
  idx_time = mdeim_idx(get_basis_time(rb_space))
  idx_space,idx_time
end

function mdeim_idx(mat::AbstractMatrix{Float})
  n = size(mat)[2]
  idx = allocate_matrix(Matrix{Float},Int,size(mat,2))
  idx[1] = Int(argmax(abs.(mat[:,1])))

  @inbounds for i = 2:n
    res = (mat[:,i] - mat[:,1:i-1] *
      (mat[idx[1:i-1],1:i-1] \ mat[idx[1:i-1],i]))
    idx[i] = Int(argmax(abs.(res)))
  end

  unique(idx)
end

function get_rb_lu(
  ::RBInfoSteady,
  rb_space::RBSpaceSteady,
  idx_space::Vector{Int})

  basis = get_basis_space(rb_space)
  get_rb_lu(basis,idx_space)
end

function get_rb_lu(info::RBInfoUnsteady,args...)
  get_rb_lu(Val(info.st_mdeim),args...)
end

function get_rb_lu(
  ::Val{false},
  rb_space::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  basis = get_basis_space(rb_space)
  get_rb_lu(basis,first(idx_st))
end

function get_rb_lu(
  ::Val{true},
  rb_space::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  basis = get_basis_space(rb_space),get_basis_time(rb_space)
  get_rb_lu(basis,idx_st)
end

function get_rb_lu(
  basis::AbstractMatrix{Float},
  idx::Vector{Int})

  basis_idx = basis[idx,:]
  lu(basis_idx)
end

function get_rb_lu(
  basis::NTuple{2,AbstractMatrix{Float}},
  idx::NTuple{2,Vector{Int}})

  bs,bt = basis
  idx_space,idx_time = idx
  bs_idx = bs[idx_space,:]
  bt_idx = bt[idx_time,:]
  bst_idx = kron(bt_idx,bs_idx)

  lu(bst_idx)
end

recast_in_full_dim(idx_tmp::Vector{Int},findnz_idx::Vector{Int}) =
  findnz_idx[idx_tmp]

recast_in_full_dim(idx_tmp::NTuple{2,Vector{Int}},findnz_idx::Vector{Int}) =
  recast_in_full_dim(first(idx_tmp),findnz_idx),last(idx_tmp)

function get_red_measure(
  op::RBVariable,
  idx::NTuple{2,Vector{Int}},
  meas::Measure)

  get_red_measure(op,first(idx),meas)
end

function get_red_measure(
  op::RBVariable,
  idx::Vector{Int},
  meas::Measure)

  get_red_measure(op,idx,get_triangulation(meas))
end

function get_red_measure(
  op::RBVariable,
  idx::Vector{Int},
  trian::Triangulation)

  el = find_mesh_elements(op,idx,trian)
  red_trian = view(trian,el)
  Measure(red_trian,get_degree(get_test(op)))
end

function find_mesh_elements(
  op::RBVariable,
  idx_tmp::Vector{Int},
  trian::Triangulation)

  idx = recast_in_mat_form(op,idx_tmp)
  connectivity = get_cell_dof_ids(op,trian)
  find_mesh_elements(Val{length(idx)>length(connectivity)}(),idx,connectivity)
end

function find_mesh_elements(::Val{true},idx::Vector{Int},connectivity)
  el = Int[]
  for eli = eachindex(connectivity)
    if !isempty(intersect(idx,abs.(connectivity[eli])))
      append!(el,eli)
    end
  end

  unique(el)
end

function find_mesh_elements(::Val{false},idx::Vector{Int},connectivity)
  el = Vector{Int}[]
  for i = idx
    eli = findall(x->!isempty(intersect(abs.(x),i)),connectivity)
    el = isempty(eli) ? el : push!(el,eli)
  end

  unique(reduce(vcat,el))
end

function recast_in_mat_form(op::RBVariable,idx_tmp::Vector{Int})
  Ns = get_Ns(get_rbspace_row(op))
  idx_space,_ = from_vec_to_mat_idx(idx_tmp,Ns)
  idx_space
end

function save(path::String,mdeim::MDEIMSteady)
  save(path,get_rbspace(mdeim))
  rb_lu = get_rb_lu(mdeim)
  save(joinpath(path,"LU"),rb_lu.factors)
  save(joinpath(path,"p"),rb_lu.ipiv)
end

function save(path::String,mdeim::MDEIMUnsteady)
  save(path,get_rbspace(mdeim))
  save(joinpath(path,"idx_space"),get_idx_space(mdeim))
  save(joinpath(path,"idx_time"),get_idx_time(mdeim))
  rb_lu = get_rb_lu(mdeim)
  save(joinpath(path,"LU"),rb_lu.factors)
  save(joinpath(path,"p"),rb_lu.ipiv)
end

function load(
  info::RBInfoSteady,
  path_id::String,
  op::RBSteadyVariable,
  meas::Measure)

  rb_space = load(info,get_id(op))
  idx_space = load(Vector{Int},joinpath(path_id,"idx_space"))
  factors = load(joinpath(path_id,"LU"))
  ipiv = load(Vector{Int},joinpath(path_id,"p"))
  rb_lu = LU(factors,ipiv,0)
  red_measure = get_red_measure(op,idx_space,meas)

  MDEIM(rb_space,rb_lu,idx_space,red_measure)
end

function load(
  info::RBInfoUnsteady,
  path_id::String,
  op::RBUnsteadyVariable,
  meas::Measure)

  rb_space = load(info,get_id(op))
  idx_space = load(Vector{Int},joinpath(path_id,"idx_space"))
  idx_time = load(Vector{Int},joinpath(path_id,"idx_time"))
  idx = (idx_space,idx_time)
  factors = load(joinpath(path_id,"LU"))
  ipiv = load(Vector{Int},joinpath(path_id,"p"))
  rb_lu = LU(factors,ipiv,0)
  red_measure = get_red_measure(op,idx_space,meas)

  MDEIM(rb_space,rb_lu,idx,red_measure)
end
