function mdeim_basis(info::RBInfo,op::RBVariable,args...)
  id = get_id(op)
  nsnap = info.mdeim_nsnap
  printstyled("MDEIM: generating $nsnap snapshots for $id \n";color=:blue)
  mdeim_basis(Val(info.fun_mdeim),info,op,args...)
end

function mdeim_basis(
  ::Val{false},
  info::RBInfoSteady,
  op::RBVariable,
  μ::Vector{Param},
  args...)

  snaps,findnz_idx = assemble_fe_snaps(op,μ,args...)
  RBSpaceSteady(snaps;ϵ=info.ϵ,style=ReducedPOD()),findnz_idx
end

function mdeim_basis(
  ::Val{false},
  info::RBInfoUnsteady,
  op::RBVariable,
  μ::Vector{Param},
  args...)

  times = get_times(op)
  snaps,findnz_idx = assemble_fe_snaps(op,μ,times,args...)
  RBSpaceUnsteady(snaps;ϵ=info.ϵ,style=ReducedPOD()),findnz_idx
end

function mdeim_basis(
  ::Val{true},
  info::RBInfoSteady,
  op::RBVariable,
  μ::Vector{Param},
  args...)

  param_snaps = assemble_functional_snaps(op,μ,args...)
  param_basis = RBSpaceSteady(param_snaps;ϵ=info.ϵ,style=ReducedPOD())
  param_fun = interpolate_param_basis(op,param_basis)
  snaps,findnz_idx = assemble_fe_snaps(op,μ,param_fun)

  RBSpaceSteady(snaps;ϵ=info.ϵ,style=ReducedPOD()),findnz_idx
end

function mdeim_basis(
  ::Val{true},
  info::RBInfo,
  op::RBUnsteadyBilinVariable,
  μ::Vector{Param},
  args...)

  param_snaps = assemble_functional_snaps(op,μ,args...)
  param_basis = RBSpaceUnsteady(param_snaps;ϵ=info.ϵ,style=ReducedPOD())
  param_fun = interpolate_param_basis(op,param_basis)
  snaps,findnz_idx = assemble_fe_snaps(op,param_fun;fun_mdeim=true)
  basis_time = get_basis_time(param_basis)
  basis_space = POD(snaps;ϵ=info.ϵ,style=ReducedPOD())

  RBSpaceUnsteady(get_id(op),basis_space,basis_time),findnz_idx
end

function interpolate_param_basis(
  op::RBVariable,
  rb_space::RBSpace)

  ns = get_ns(rb_space)
  basis_space = get_basis_space(rb_space)
  test = get_test(op)
  quad_fespace = LagrangianQuadFESpace(test)
  quad_test = quad_fespace.test

  [FEFunction(quad_test,basis_space[:,k]) for k = 1:ns]
end

function interpolate_param_basis(
  op::RBVariable{Nonlinear,Top},
  rb_space::RBSpace) where Top

  Ns,ns = get_Ns(rb_space),get_ns(rb_space)
  test = get_test(op)
  basis_space = get_basis_space(rb_space)
  basis_free,basis_dir = basis_space[1:Ns,:],basis_space[1+Ns:end,:]

  [FEFunction(test,basis_free[:,k],basis_dir[:,k]) for k = 1:ns]
end
