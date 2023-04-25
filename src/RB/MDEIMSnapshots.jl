function mdeim_basis(info::RBInfo,op::RBVariable,args...)
  state = info.fun_mdeim && typeof(op) == RBBilinVariable
  mdeim_basis(Val(state),info,op,args...)
end

function mdeim_basis(
  ::Val{false},
  info::RBInfoSteady,
  op::RBVariable,
  μ::Vector{Param},
  args...)

  da = DistributedAssembler(op,μ,args...)
  snaps,findnz_idx = assemble(da,μ)
  RBSpaceSteady(snaps;ϵ=info.ϵ,style=ReducedPOD()),findnz_idx
end

function mdeim_basis(
  ::Val{false},
  info::RBInfoUnsteady,
  op::RBVariable,
  μ::Vector{Param},
  args...)

  da = DistributedAssembler(op,μ,args...)
  snaps,findnz_idx = assemble(da,μ)
  RBSpaceUnsteady(snaps;ϵ=info.ϵ,style=ReducedPOD()),findnz_idx
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

  #findnz_idx = (op;μ,u)

  RBSpaceUnsteady(id,bs,param_bt),findnz_idx
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

  times = get_times(op)
  eval_param(k,t) = Broadcasting(x->param_fun(x,μ[k],t))(quadp)
  eval_param(k) = Matrix([eval_param(k,t) for t = times])
  eval_param
end

function reduce_param_function(
  op::RBUnsteadyVariable,
  vals::Matrix{Float})

  param_bs = POD(vals)
  vals2 = mode2_unfolding(param_bs'*vals,Int(size(vals,2)/get_Nt(op)))
  param_bt = POD(vals2)
  param_bs,param_bt
end

function reduce_param_function(
  op::RBUnsteadyVariable{Nonlinear,Ttr},
  all_vals::Matrix{Float}) where Ttr

  Ns = get_Ns(get_rbspace_row(op))
  Nt = get_Nt(op)
  ns = Int(size(free_vals,2)/Nt)
  free_vals = all_vals[1:Ns,:]

  bs_all = POD(all_vals;ϵ=info.ϵ)
  bs_free = bs_all[1:Ns,:]
  s2 = mode2_unfolding(bs_free'*free_vals,ns)
  bt_free = POD(s2;ϵ=info.ϵ)

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
