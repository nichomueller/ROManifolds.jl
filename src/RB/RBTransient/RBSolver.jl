struct MDEIMCombineReduction{A,R<:AbstractReduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
  reduction::R
  combine::Function
  online_nsnaps::Int
end

function RBSteady.MDEIMReduction(
  combine::Function;
  red_style=SearchSVDRank(1e-4),
  nsnaps=20,
  online_nsnaps=10,
  reduction=PODReduction(;red_style,nsnaps))

  MDEIMCombineReduction(reduction,combine,online_nsnaps)
end

function RBSteady.MDEIMReduction(;combine=(x,y)->x,kwargs...)
  MDEIMReduction(combine;kwargs...)
end

RBSteady.get_reduction(r::MDEIMCombineReduction) = get_reduction(r.reduction)
RBSteady.ReductionStyle(r::MDEIMCombineReduction) = ReductionStyle(get_reduction(r))
RBSteady.NormStyle(r::MDEIMCombineReduction) = NormStyle(get_reduction(r))
RBSteady.num_snaps(r::MDEIMCombineReduction) = num_snaps(get_reduction(r))
RBSteady.num_online_snaps(r::MDEIMCombineReduction) = r.online_nsnaps

function RBSteady.RBSolver(fesolver::ODESolver,args...;kwargs...)
  @notimplemented
end

function RBSteady.RBSolver(
  fesolver::ThetaMethod;
  state_reduction=PODReduction(),
  residual_reduction=MDEIMReduction(),
  kwargs...)

  θ = fesolver.θ
  combine_jac = (x,y) -> θ*x+(1-θ)*y
  combine_djacdt = (x,y) -> θ*(x-y)
  jacobian_reduction = (MDEIMReduction(combine_jac),MDEIMReduction(combine_djacdt))
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function RBSolver(
  fesolver::ThetaMethod;
  red_style=SearchSVDRank(1e-4),
  norm_style=EuclideanNorm(),
  state_nsnaps=50,
  res_nsnaps=20,
  jac_snaps=20)

  state_reduction = PODReduction(red_style,norm_style,state_nsnaps)
  residual_reduction = MDEIMReduction(red_style,res_nsnaps)
  θ = fesolver.θ
  combine_jac = (x,y) -> θ*x+(1-θ)*y
  combine_djacdt = (x,y) -> θ*(x-y)
  jacobian_reduction = (MDEIMReduction(combine_jac),MDEIMReduction(combine_djacdt))
  jacobian_reduction = MDEIMReduction(red_style,jac_snaps)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function RBSteady.fe_solutions(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  nparams=num_params(solver),
  r=realization(op;nparams))

  fesolver = get_fe_solver(solver)
  timer = get_timer(solver)
  reset_timer!(timer)

  sol = solve(fesolver,op,uh0,timer;r)
  odesol = sol.odesol
  r = odesol.r

  values = collect(sol)

  i = get_vector_index_map(op)
  snaps = Snapshots(values,i,r)
  return snaps
end
