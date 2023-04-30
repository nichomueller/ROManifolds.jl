include("MDEIMSnapshots.jl")

abstract type MDEIM end

struct MDEIMSteady <: MDEIM
  rbspace::RBSpaceSteady
  rb_lu::EMatrix{Float}
  idx::Vector{Int}
  red_measure::Measure
end

struct MDEIMUnsteady <: MDEIM
  rbspace::RBSpaceUnsteady
  rb_lu::EMatrix{Float}
  idx::NTuple{2,Vector{Int}}
  red_measure::Measure
end

function MDEIM(
  red_rbspace::RBSpaceSteady,
  rb_lu::EMatrix{Float},
  idx::Vector{Int},
  red_measure::Measure)

  MDEIMSteady(red_rbspace,rb_lu,idx,red_measure)
end

function MDEIM(
  red_rbspace::RBSpaceUnsteady,
  rb_lu::EMatrix{Float},
  idx::NTuple{2,Vector{Int}},
  red_measure::Measure)

  MDEIMUnsteady(red_rbspace,rb_lu,idx,red_measure)
end

function MDEIM(
  info::RBInfo,
  op::RBVariable,
  measures::ProblemMeasures,
  field::Symbol,
  μ::Vector{Param},
  args...)

  μ_mdeim = μ[1:info.mdeim_nsnap]
  meas = getproperty(measures,field)

  rbspace,findnz_idx = mdeim_basis(info,op,μ_mdeim,args...)
  red_rbspace = project_mdeim_basis(op,rbspace,findnz_idx)
  idx = mdeim_idx(rbspace)
  rb_lu = get_rb_lu(info,rbspace,idx)
  idx = recast_in_full_dim(idx,findnz_idx)
  red_meas = get_red_measure(op,idx,meas)

  MDEIM(red_rbspace,rb_lu,idx,red_meas)
end

get_rbspace(mdeim::MDEIM) = mdeim.rbspace

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
  rbspace::RBSpace,
  args...)

  id = get_id(rbspace)
  bs = rb_space_projection(op,rbspace,args...)
  RBSpaceSteady(id,bs)
end

function project_mdeim_basis(
  op::RBUnsteadyVariable,
  rbspace::RBSpace,
  args...)

  id = get_id(rbspace)
  bs = rb_space_projection(op,rbspace,args...)
  bt = get_basis_time(rbspace)
  RBSpaceUnsteady(id,bs,bt)
end

function rb_space_projection(
  op::RBVariable,
  rbspace::RBSpace,
  args...)

  rb_space_projection(op,get_basis_space(rbspace),args...)
end

function rb_space_projection(
  op::RBLinVariable,
  basis_space::AbstractMatrix{Float},
  args...)

  rbspace_row = get_rbspace_row(op)
  brow = get_basis_space(rbspace_row)
  brow'*basis_space
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
  red_basis_space = zeros(get_ns(rbspace_row)*get_ns(rbspace_col),Qs)
  for q = 1:Qs
    vsparse = basis_space[:,q]
    smat = sparse(isparse,jsparse,vsparse,Ns,Ns)
    red_basis_space[:,q] = Vector(brow'*smat*bcol)
  end

  red_basis_space
end

function mdeim_idx(rbspace::RBSpaceSteady)
  idx_space = mdeim_idx(get_basis_space(rbspace))
  idx_space
end

function mdeim_idx(rbspace::RBSpaceUnsteady)
  idx_space = mdeim_idx(get_basis_space(rbspace))
  idx_time = mdeim_idx(get_basis_time(rbspace))
  idx_space,idx_time
end

function mdeim_idx(mat::AbstractMatrix{Float})
  n = size(mat)[2]
  idx = zeros(Int,size(mat,2))
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
  rbspace::RBSpaceSteady,
  idx_space::Vector{Int})

  basis = get_basis_space(rbspace)
  get_rb_lu(basis,idx_space)
end

function get_rb_lu(info::RBInfoUnsteady,args...)
  get_rb_lu(Val(info.st_mdeim),args...)
end

function get_rb_lu(
  ::Val{false},
  rbspace::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  basis = get_basis_space(rbspace)
  get_rb_lu(basis,first(idx_st))
end

function get_rb_lu(
  ::Val{true},
  rbspace::RBSpaceUnsteady,
  idx_st::NTuple{2,Vector{Int}})

  basis = get_basis_space(rbspace),get_basis_time(rbspace)
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
  bst_idx = EMatrix(kron(bt_idx,bs_idx))

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

recast_in_mat_form(::RBLinVariable,idx_tmp::Vector{Int}) = idx_tmp

function recast_in_mat_form(op::RBBilinVariable,idx_tmp::Vector{Int})
  Ns = get_Ns(get_rbspace_row(op))
  idx_space,_ = from_vec_to_mat_idx(idx_tmp,Ns)
  idx_space
end

function save(path::String,mdeim::MDEIMSteady)
  save(path,get_rbspace(mdeim))
  save(joinpath(path,"rb_lu"),get_rb_lu(mdeim))
end

function save(path::String,mdeim::MDEIMUnsteady)
  save(path,get_rbspace(mdeim))
  save(joinpath(path,"idx_space"),get_idx_space(mdeim))
  save(joinpath(path,"idx_time"),get_idx_time(mdeim))
  save(joinpath(path,"rb_lu"),get_rb_lu(mdeim))
end

function load(
  info::RBInfoSteady,
  op::RBSteadyVariable,
  meas::Measure)

  id = get_id(op)
  path_id = joinpath(info.offline_path,"$id")

  rbspace = load(info,id)
  idx_space = load(Vector{Int},joinpath(path_id,"idx_space"))
  rb_lu = load(EMatrix{Float},joinpath(path_id,"rb_lu"))
  red_measure = get_red_measure(op,idx_space,meas)

  MDEIM(rbspace,rb_lu,idx_space,red_measure)
end

function load(
  info::RBInfoUnsteady,
  op::RBUnsteadyVariable,
  meas::Measure)

  id = get_id(op)
  path_id = joinpath(info.offline_path,"$id")

  rbspace = load(info,id)
  idx_space = load(Vector{Int},joinpath(path_id,"idx_space"))
  idx_time = load(Vector{Int},joinpath(path_id,"idx_time"))
  idx = (idx_space,idx_time)
  rb_lu = load(EMatrix{Float},joinpath(path_id,"rb_lu"))
  red_measure = get_red_measure(op,idx_space,meas)

  MDEIM(rbspace,rb_lu,idx,red_measure)
end
