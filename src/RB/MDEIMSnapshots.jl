function vector_snapshots(
  op::RBLinVariable{Nonaffine},
  μ::Vector{Param})

  id = get_id(op)
  printstyled("MDEIM: generating snapshots for $id \n";color=:blue)

  #MyFunc = whatever(ProblemDescription)
  #push!(vals, MyFunc(V, k))
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

  ns = size(param_bs,2)
  vals = Vector{Float}[]
  @threads for k = 1:ns
    push!(vals,snapshot(k))
  end
  bs = rb_space(info,Snapshots(id,vals))

  findnz_map,RBSpaceUnsteady(id,bs,param_bt)
end

function matrix_snapshots(
  info::RBInfo,
  op::RBUnsteadyBilinVariable{Nonlinear,<:ParamTransientTrialFESpace},
  μ::Vector{Param},
  ugh::Snapshots)

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

  Ns = get_Ns(get_rbspace_row(op))
  bs_ug = reduced_POD(get_snap(ugh);ϵ=info.ϵ)
  bs_u,bs_g = bs_ug[1:Ns,:],bs_ug[1+Ns:end,:]
  s2 = mode2_unfolding(bs_u'*get_snap(ugh)[1:Ns,:],get_nsnap(ugh))
  bt_u = reduced_POD(s2;ϵ=info.ϵ)
  bst_u_fun = interpolate_param_function(op,bs_u,bs_g)

  findnz_map = get_findnz_map(op,μ,bst_u_fun)
  M = assemble_functional_variable(op)

  function snapshot(k::Int)
    b = bst_u_fun(k)
    nonzero_values(M(b,first(μ),0.),findnz_map)
  end

  ns = size(bs_u,2)
  vals = Vector{Float}[]
  @threads for k = 1:ns
    push!(vals,snapshot(k))
  end
  bs = rb_space(info,Snapshots(id,vals))

  findnz_map,RBSpaceUnsteady(id,bs,bt_u)
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
  bsuθ::Matrix{Float},
  bsgθ::Matrix{Float})::Function where Top

  test = get_test(op)
  param_fun(k::Int) = FEFunction(test,bsuθ[:,k],bsgθ[:,k])
  param_fun
end
