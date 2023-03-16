function vector_snapshots(
  op::RBLinVariable{Nonaffine},
  μ::Vector{Param})

  id = get_id(op)
  printstyled("MDEIM: generating snapshots for $id \n";color=:blue)

  V = assemble_vector(op)
  vals = Array{Float}[]
  @threads for k in eachindex(μ)
    push!(vals,V(μ[k]))
  end

  Snapshots(id,vals)
end

function vector_snapshots(
  op::RBLinVariable{Nonlinear},
  μ::Vector{Param},
  uh::Snapshots)

  id = get_id(op)
  printstyled("MDEIM: generating snapshots for $id \n";color=:blue)

  u_fun(k::Int) = FEFunction(op,uh[k],μ[k])
  V = assemble_vector(op)
  vals = Array{Float}[]
  @threads for k in eachindex(μ)
    push!(vals,V(μ[k],u_fun(k)))
  end

  Snapshots(id,vals)
end

#= function vector_snapshots(
  op::RBLinVariable,
  μ::Vector{Param})

  id = get_id(op)
  printstyled("MDEIM: generating snapshots on the quadrature points, $id \n";
   color=:blue)

  param_vals = evaluate_param_function(op,μ)
  param_bs,param_bt = reduce_param_function(op,param_vals)

  param_fun = interpolate_param_function(op,param_bs)

  V = assemble_functional_variable(op)

  function snapshot(k::Int)::Vector{Float}
    b = param_fun(k)
    V(b,μ[k],0.)
  end

  ns,nt = size(param_bs,2),size(param_bt,2)
  vals = Vector{Float}[]
  @threads for k = 1:ns
    push!(vals,snapshot(k))
  end

  vals_st = [reshape(kron(param_bt[:,time_idx(k,ns)],vals[space_idx(k,ns)]),:,
    get_Nt(op)) for k = 1:ns*nt]

  Snapshots(id,vals_st)
end =#

#= function vector_snapshots(
  op::RBLinVariable,
  μ::Vector{Param},
  rbspace_uθ::RBSpaceUnsteady)

  id = get_id(op)
  printstyled("MDEIM: generating snapshots on the quadrature points, $id \n";
   color=:blue)

  param_bs,param_bt = get_basis_space(rbspace_uθ),get_basis_time(rbspace_uθ)

  param_fun = interpolate_param_function(op,param_bs)
  V = assemble_functional_variable(op)

  function snapshot(k::Int)::Vector{Float}
    b = param_fun(k)
    V(b,μ[k],0.)
  end

  ns,nt = size(param_bs,2),size(param_bt,2)
  vals = Vector{Float}[]
  @threads for k = 1:ns
    push!(vals,snapshot(k))
  end

  vals_st = [reshape(kron(param_bt[:,time_idx(k,ns)],vals[space_idx(k,ns)]),:,
    get_Nt(op)) for k = 1:ns*nt]

  Snapshots(id,vals_st)
end =#

function matrix_snapshots(
  op::RBBilinVariable,
  μ::Vector{Param})

  id = get_id(op)
  printstyled("MDEIM: generating snapshots for $id \n";color=:blue)

  findnz_map = get_findnz_map(op,μ)
  M = assemble_matrix(op)
  vals = Matrix{Float}[]
  @threads for k in eachindex(μ)
    push!(vals,nonzero_values(M(μ[k]),findnz_map))
  end

  findnz_map,Snapshots(id,vals)
end

function matrix_snapshots(
  op::RBBilinVariable{Nonlinear,Ttr},
  μ::Vector{Param},
  uh::Snapshots) where Ttr

  id = get_id(op)
  printstyled("MDEIM: generating snapshots for $id \n";color=:blue)

  u_fun(k::Int) = FEFunction(op,uh[k],μ[k])
  findnz_map = get_findnz_map(op,μ,u_fun)
  M = assemble_matrix(op)
  vals = Matrix{Float}[]
  @threads for k in eachindex(μ)
    push!(vals,nonzero_values(M(μ[k],u_fun(k)),findnz_map))
  end

  findnz_map,Snapshots(id,vals)
end

function matrix_snapshots(
  info::RBInfo,
  op::RBUnsteadyBilinVariable{Top,<:ParamTransientTrialFESpace},
  μ::Vector{Param}) where Top

  id = get_id(op)
  printstyled("MDEIM: generating snapshots on the quadrature points, $id \n";
   color=:blue)

  param_vals = evaluate_param_function(op,μ)
  param_bs,param_bt = reduce_param_function(op,param_vals)
  param_fun = interpolate_param_function(op,param_bs)

  findnz_map = get_findnz_map(op,μ)
  M = assemble_functional_variable(op)

  function snapshot(k::Int)
    b = param_fun(k)
    nonzero_values(M(b,μ[k],0.),findnz_map)
  end

  #ns,nt = size(param_bs,2),size(param_bt,2)
  ns = size(param_bs,2)
  vals = Vector{Float}[]
  @threads for k = 1:ns
    push!(vals,snapshot(k))
  end
  bs = rb_space(info,Snapshots(id,vals))

  #= vals_st = [reshape(kron(param_bt[:,time_idx(k,ns)],vals[space_idx(k,ns)]),:,
    get_Nt(op)) for k = 1:ns*nt] =#

  findnz_map,RBSpaceUnsteady(id,bs,param_bt)#Snapshots(id,vals_st)
end

function matrix_snapshots(
  info::RBInfo,
  op::RBUnsteadyBilinVariable{Nonlinear,<:ParamTransientTrialFESpace},
  μ::Vector{Param},
  uh::Snapshots)

  id = get_id(op)
  printstyled("MDEIM: generating snapshots on the quadrature points, $id \n";
    color=:blue)

  #REMOVE THIS WHEN POSSIBLE!
  function get_findnz_map(
    op::RBUnsteadyVariable{Nonlinear,Ttr},
    μ::Vector{Param},
    f::Function)::Vector{Int} where Ttr

    dtθ = get_dt(op)*get_θ(op)
    M = assemble_matrix(op,dtθ)(first(μ),f(1))
    first(findnz(M[:]))
  end

  bst_u = rb(info,uh)
  bs_u,bt_u = get_basis_space(bst_u),get_basis_time(bst_u)
  bst_u_fun = interpolate_param_function(op,bst_u)

  findnz_map = get_findnz_map(op,μ,bst_u_fun)
  M = assemble_functional_variable(op)

  function snapshot(k::Int)
    b = bst_u_fun(k)
    nonzero_values(M(b,first(μ),0.),findnz_map)
  end

  #ns,nt = size(bs_u,2),size(bt_u,2)
  ns = size(bs_u,2)
  vals = Vector{Float}[]
  @threads for k = 1:ns
    push!(vals,snapshot(k))
  end
  bs = rb_space(info,Snapshots(id,vals))
  #= red_vals = rb_space_projection(op,bs,findnz_map)

  vals_st = Matrix{Float}[]
  @threads for k = 1:ns*nt
    val_k = reshape(kron(bt_u[:,time_idx(k,ns)],red_vals[:,space_idx(k,ns)]),:,get_Nt(op))
    push!(vals_st,val_k)
  end =#

  findnz_map,RBSpaceUnsteady(id,bs,bt_u)#Snapshots(id,vals_st)
end

function evaluate_param_function(
  op::RBUnsteadyVariable,
  μ::Vector{Param})::Matrix{Float}

  timesθ = get_timesθ(op)
  phys_cells = get_phys_quad_points(op)
  ncells = length(phys_cells)
  nquad_cell = length(first(phys_cells))
  nquadp = ncells*nquad_cell
  dim = get_dimension(op)
  quadp = zeros(VectorValue{dim,Float},nquadp)
  for (i,celli) = enumerate(phys_cells)
    quadp[(i-1)*nquad_cell+1:i*nquad_cell] = celli
  end

  param_fun = get_param_function(op)

  param(μ,t) = Broadcasting(x->param_fun(x,μ,t))(quadp)
  param(μ) = Matrix([param(μ,t) for t = timesθ])
  param_vals = Matrix{Float}[]
  @threads for μk = μ
    push!(param_vals,param(μk))
  end

  Matrix(param_vals)
end

function reduce_param_function(
  op::RBUnsteadyVariable,
  vals::Matrix{Float})::NTuple{2,Matrix{Float}}

  param_bs = reduced_POD(vals)
  vals2 = mode2_unfolding(param_bs'*vals,Int(size(vals,2)/get_Nt(op)))
  param_bt = reduced_POD(vals2)
  param_bs,param_bt
end

function interpolate_param_function(
  op::RBUnsteadyVariable,
  vals::Matrix{Float})::Function

  test_quad = LagrangianQuadFESpace(get_test(op))
  param_fun = FEFunction(test_quad,vals)
  param_fun
end

function interpolate_param_function(
  op::RBUnsteadyVariable{Nonlinear,Top},
  rbspace_uθ::RBSpaceUnsteady)::Function where Top

  bsθ = get_basis_space(rbspace_uθ)
  test = get_test(op)
  param_fun(k::Int) = FEFunction(test,bsθ[:,k])
  param_fun
end
