struct TransientPODReduction{A,B} <: AbstractPODReduction{A,B}
  reduction_space::PODReduction{A,B}
  reduction_time::PODReduction{A,EuclideanNorm}
end

function TransientPODReduction(
  tolrank_space::Union{Int,Float64},
  tolrank_time::Union{Int,Float64},
  args...;kwargs...)

  reduction_space = PODReduction(tolrank_space,args...;kwargs...)
  reduction_time = PODReduction(tolrank_time;kwargs...)
  TransientPODReduction(reduction_space,reduction_time)
end

function TransientPODReduction(tolrank::Union{Int,Float64},args...;kwargs...)
  TransientPODReduction(tolrank,tolrank,args...;kwargs...)
end

get_reduction_space(r::TransientPODReduction) = RBSteady.get_reduction(r.reduction_space)
get_reduction_time(r::TransientPODReduction) = RBSteady.get_reduction(r.reduction_time)
RBSteady.ReductionStyle(r::TransientPODReduction) = ReductionStyle(get_reduction_space(r))
RBSteady.NormStyle(r::TransientPODReduction) = NormStyle(get_reduction_space(r))
ParamDataStructures.num_params(r::TransientPODReduction) = num_params(get_reduction_space(r))

# generic constructor

function TransientReduction(red_style::Union{SearchSVDRank,FixedSVDRank},args...;kwargs...)
  reduction_space = PODReduction(red_style,args...;kwargs...)
  reduction_time = PODReduction(red_style;kwargs...)
  TransientPODReduction(reduction_space,reduction_time)
end

function TransientReduction(red_style::Union{SearchTTSVDRanks,FixedTTSVDRanks},args...;kwargs...)
  TTSVDReduction(red_style,args...;kwargs...)
end

struct TransientMDEIMReduction{A,R<:AbstractReduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
  reduction::R
  combine::Function
  nparams_test::Int
end

function TransientMDEIMReduction(combine::Function,args...;nparams_test=10,kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  TransientMDEIMReduction(reduction,combine,nparams_test)
end

RBSteady.get_reduction(r::TransientMDEIMReduction) = RBSteady.get_reduction(r.reduction)
RBSteady.ReductionStyle(r::TransientMDEIMReduction) = ReductionStyle(RBSteady.get_reduction(r))
RBSteady.NormStyle(r::TransientMDEIMReduction) = NormStyle(RBSteady.get_reduction(r))
ParamDataStructures.num_params(r::TransientMDEIMReduction) = num_params(RBSteady.get_reduction(r))
RBSteady.num_online_params(r::TransientMDEIMReduction) = r.nparams_test

function RBSteady.RBSolver(fesolver::ODESolver,state_reduction::AbstractReduction;kwargs...)
  @notimplemented
end

function RBSteady.RBSolver(
  fesolver::ThetaMethod,
  state_reduction::AbstractReduction;
  nparams_res=20,
  nparams_jac=20,
  nparams_djac=nparams_jac,
  nparams_test=10)

  θ = fesolver.θ
  combine_res = (x) -> x
  combine_jac = (x,y) -> θ*x+(1-θ)*y
  combine_djac = (x,y) -> θ*(x-y)

  red_style = ReductionStyle(state_reduction)

  residual_reduction = TransientMDEIMReduction(combine_res,red_style;nparams=nparams_res,nparams_test)
  jac_reduction = TransientMDEIMReduction(combine_jac,red_style;nparams=nparams_jac,nparams_test)
  djac_reduction = TransientMDEIMReduction(combine_djac,red_style;nparams=nparams_djac,nparams_test)
  jacobian_reduction = (jac_reduction,djac_reduction)

  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

RBSteady.num_jac_params(s::RBSolver{<:ODESolver}) = num_params(first(s.jacobian_reduction))

function RBSteady.fe_solutions(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  nparams=num_params(solver),
  r=realization(op;nparams))

  fesolver = get_fe_solver(solver)

  sol = solve(fesolver,op,uh0;r)
  odesol = sol.odesol
  r = odesol.r

  values,stats = collect(sol)

  i = get_vector_index_map(op)
  snaps = Snapshots(values,i,r)
  return snaps,stats
end
