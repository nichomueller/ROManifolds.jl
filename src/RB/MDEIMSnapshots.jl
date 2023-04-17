function mdeim_basis(info::RBInfo,op::RBVariable,args...)
  state = info.fun_mdeim && typeof(op) == RBBilinVariable
  mdeim_basis(Val(state),info,op,args...)
end

function mdeim_basis(::Val{false},info::RBInfoSteady,op::RBVariable,args...)
  snaps = fe_snapshots(op,args...)
  id = get_id(snaps)
  basis_space = mdeim_POD(snaps;ϵ=info.ϵ)
  RBSpaceSteady(id,basis_space)
end

function mdeim_basis(::Val{false},info::RBInfoUnsteady,op::RBVariable,args...)
  snaps = fe_snapshots(op,args...)
  id = get_id(snaps)
  s,ns = get_snap(snaps),get_nsnap(snaps)
  basis_space = reduced_POD(s;ϵ=info.ϵ)
  s2 = mode2_unfolding(basis_space'*s,ns)
  basis_time = reduced_POD(s2;ϵ=info.ϵ)
  RBSpaceUnsteady(id,basis_space,basis_time)
end

function fe_snapshots(op::RBVariable,args...)
  id = get_id(op)
  printstyled("MDEIM: generating snapshots for $id \n";color=:blue)

  fe_quantity = get_assembler(op,args...)
  vals = Vector{Float}[]
  @threads for k in eachindex(μ)
    push!(vals,fe_quantity(k))
  end

  findnz_map = get_findnz_map(op;args...)

  Snapshots(id,vals),findnz_map
end

function mdeim_basis(
  ::Val{true},
  info::RBInfo,
  op::RBUnsteadyBilinVariable{Top,<:ParamTransientTrialFESpace},
  μ::Vector{Param},
  u=nothing) where Top

  id = get_id(op)
  printstyled("MDEIM: generating snapshots on the quadrature points, $id \n";
   color=:blue)

  param_vals = evaluate_param_function(op,μ,u)
  param_bs,param_bt = reduce_param_function(op,param_vals)
  param_fun = interpolate_param_function(op,param_bs)

  fe_quantity = get_assembler(op,μ,param_fun)
  ns = size(param_bs,2)
  vals = Vector{Float}[]
  @threads for k = 1:ns
    push!(vals,fe_quantity(k))
  end
  bs = rb_space(info,Snapshots(id,vals))

  findnz_map = get_findnz_map(op;μ,u)

  RBSpaceUnsteady(id,bs,param_bt),findnz_map
end

function get_assembler(
  op::RBVariable{Nonaffine,Ttr},
  μ::Vector{Param}) where Ttr

  k -> assemble_vector(op;μ=μ[k])
end

function get_assembler(
  op::RBVariable{Nonlinear,Ttr},
  μ::Vector{Param},
  uh::Snapshots) where Ttr

  u_fun(k) = FEFunction(op,uh[k],μ[k])
  k -> assemble_vector(op;μ=μ[k],u=u_fun(k))
end

function get_assembler(
  op::RBBilinVariable{Nonaffine,Ttr},
  μ::Vector{Param},
  fun::Function) where Ttr

  k -> assemble_vector(op;μ=μ[k],u=fun(k),t=first(get_timesθ(op)))
end

function get_assembler(
  op::RBBilinVariable{Nonlinear,Ttr},
  μ::Vector{Param},
  fun::Function) where Ttr

  k -> assemble_vector(op;μ=first(μ),u=fun(k),t=first(get_timesθ(op)))
end

function evaluate_param_function(
  op::RBVariable,
  μ::Vector{Param},
  args...)

  param_fun = get_param_function(op)
  quadp = get_phys_quad_points(op)
  eval_fun = get_evaluated_param_function(op,param_fun,quadp,μ)

  param_vals = Array{Float}[]
  @threads for k = eachindex(μ)
    push!(param_vals,eval_fun(k))
  end

  Matrix(param_vals)
end

function evaluate_param_function(
  ::RBVariable{Nonlinear,Ttr},
  ::Vector{Param},
  vals::Snapshots) where Ttr

  get_snap(vals)
end

function get_evaluated_param_function(
  ::RBSteadyVariable,
  param_fun::Function,
  quadp::Vector{Point},
  μ::Vector{Param})

  k -> Broadcasting(x->param_fun(x,μ[k]))(quadp)
end

function get_evaluated_param_function(
  op::RBUnsteadyVariable,
  param_fun::Function,
  quadp::Vector{Point},
  μ::Vector{Param})

  timesθ = get_timesθ(op)
  eval_param(k,t) = Broadcasting(x->param_fun(x,μ[k],t))(quadp)
  eval_param(k) = Matrix([eval_param(k,t) for t = timesθ])
  eval_param
end

function reduce_param_function(
  op::RBUnsteadyVariable,
  vals::Matrix{Float})

  param_bs = reduced_POD(vals)
  vals2 = mode2_unfolding(param_bs'*vals,Int(size(vals,2)/get_Nt(op)))
  param_bt = reduced_POD(vals2)
  param_bs,param_bt
end

function reduce_param_function(
  op::RBUnsteadyVariable{Nonlinear,Ttr},
  all_vals::Matrix{Float}) where Ttr

  Ns = get_Ns(get_rbspace_row(op))
  Nt = get_Nt(op)
  ns = Int(size(free_vals,2)/Nt)
  free_vals = all_vals[1:Ns,:]

  bs_all = reduced_POD(all_vals;ϵ=info.ϵ)
  bs_free = bs_all[1:Ns,:]
  s2 = mode2_unfolding(bs_free'*free_vals,ns)
  bt_free = reduced_POD(s2;ϵ=info.ϵ)

  bs_all,bt_free
end

function interpolate_param_function(
  op::RBUnsteadyVariable,
  vals::Matrix{Float})

  test_quad = LagrangianQuadFESpace(get_test(op))
  param_fun = FEFunction(test_quad,vals)
  param_fun
end

function interpolate_param_function(
  op::RBUnsteadyVariable{Nonlinear,Top},
  vals::Matrix{Float}) where Top

  Ns = get_Ns(get_rbspace_row(op))
  vals_free,vals_dir = vals[1:Ns,:],vals[1+Ns:end,:]

  test = get_test(op)
  param_fun(k::Int) = FEFunction(test,vals_free[:,k],vals_dir[:,k])
  param_fun
end
