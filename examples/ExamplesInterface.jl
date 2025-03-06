module ExamplesInterface

using DrWatson
using Gridap
using Plots
using Serialization
using Test

using ROManifolds
using ROManifolds.RBSteady
using ROManifolds.RBTransient

import Gridap.CellData: get_domains
import Gridap.FESpaces: get_algebraic_operator
import Gridap.Helpers: @abstractmethod
import Gridap.MultiField: BlockMultiFieldStyle
import ROManifolds.ParamDataStructures: AbstractSnapshots
import ROManifolds.ParamSteady: ParamOperator,LinearNonlinearParamEq,change_domains
import ROManifolds.ParamODEs: ODEParamOperator,LinearNonlinearParamODE
import ROManifolds.RBSteady: reduced_operator,get_state_reduction,get_residual_reduction,get_jacobian_reduction

function try_loading_fe_snapshots(dir,rbsolver,feop,args...;kwargs...)
  try
    fesnaps = load_snapshots(dir)
    println("Load snapshots at $dir succeeded!")
    return fesnaps
  catch
    println("Load snapshots at $dir failed, must compute them")
    fesnaps, = solution_snapshots(rbsolver,feop,args...;kwargs...)
    save(dir,fesnaps)
    return fesnaps
  end
end

function try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps)
  try
    rbop = load_operator(dir_tol,feop)
    println("Load reduced operator at $dir_tol succeeded!")
    return rbop
  catch
    println("Load reduced operator at $dir_tol failed, must run offline phase")
    op = get_algebraic_operator(feop)
    dir = joinpath(splitpath(dir_tol)[1:end-1])
    local res,jac
    try
      res = load_residuals(dir,feop)
      jac = load_jacobians(dir,feop)
    catch
      res = residual_snapshots(rbsolver,op,fesnaps)
      jac = jacobian_snapshots(rbsolver,op,fesnaps)
      save(dir,res,feop;label="res")
      save(dir,jac,feop;label="jac")
    end
    rbop = reduced_operator(rbsolver,feop,fesnaps,jac,res)
    save(dir_tol,rbop)
    return rbop
  end
end

get_error(perf::ROMPerformance) = perf.error

function plot_errors(dir,tols,perfs::Vector{ROMPerformance})
  errs = map(get_error,perfs)
  n = length(first(errs))
  errvec = map(i -> getindex.(errs,i),1:n)
  labvec = n==1 ? "Error" : ["Error $i" for i in 1:n]

  file = joinpath(dir,"convergence.png")
  p = plot(tols,tols,lw=3,label="Tol.")
  scatter!(tols,errvec,lw=3,label=labvec)
  plot!(xscale=:log10,yscale=:log10)
  xlabel!("Tolerance")
  ylabel!("Error")
  title!("Average relative error")

  savefig(p,file)
end

update_redstyle(rs::SearchSVDRank,tol) = SearchSVDRank(tol)
update_redstyle(rs::LRApproxRank,tol) = LRApproxRank(tol)
update_redstyle(rs::TTSVDRanks,tol) = TTSVDRanks(map(s->update_redstyle(s,tol),rs.style),rs.unsafe)

function update_reduction(red::Reduction,tol)
  @abstractmethod
end

function update_reduction(red::AffineReduction,tol)
  AffineReduction(update_redstyle(red.red_style,tol),red.norm_style)
end

function update_reduction(red::PODReduction,tol)
  PODReduction(update_redstyle(red.red_style,tol),red.norm_style,red.nparams)
end

function update_reduction(red::TTSVDReduction,tol)
  TTSVDReduction(update_redstyle(red.red_style,tol),red.norm_style,red.nparams)
end

function update_reduction(red::SupremizerReduction,tol)
  SupremizerReduction(update_reduction(red.reduction,tol),red.supr_op,red.supr_tol)
end

function update_reduction(red::MDEIMReduction,tol)
  MDEIMReduction(update_reduction(red.reduction,tol))
end

function update_reduction(red::TransientReduction,tol)
  TransientReduction(
    update_reduction(red.reduction_space,tol),
    update_reduction(red.reduction_time,tol)
    )
end

function update_reduction(red::TransientMDEIMReduction,tol)
  TransientMDEIMReduction(update_reduction(red.reduction,tol),red.combine)
end

function update_reduction(red::NTuple{N,TransientMDEIMReduction},tol) where N
  map(r->update_reduction(r,tol),red)
end

function update_solver(rbsolver::RBSolver,tol)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = update_reduction(get_state_reduction(rbsolver),tol)
  residual_reduction = update_reduction(get_residual_reduction(rbsolver),tol)
  jacobian_reduction = update_reduction(get_jacobian_reduction(rbsolver),tol)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function reduced_operator(rbsolver::RBSolver,op::ParamOperator,red_trial,red_test,jac,res)
  jac_red = get_jacobian_reduction(rbsolver)
  red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jac)
  res_red = get_residual_reduction(rbsolver)
  red_rhs = reduced_residual(res_red,red_test,res)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  op′ = change_domains(op,trians_rhs,trians_lhs)
  GenericRBOperator(op′,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(rbsolver::RBSolver,odeop::ODEParamOperator,red_trial,red_test,jac,res)
  jac_red = get_jacobian_reduction(rbsolver)
  red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jac)
  res_red = get_residual_reduction(rbsolver)
  red_rhs = reduced_residual(res_red,red_test,res)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  odeop′ = change_domains(odeop,trians_rhs,trians_lhs)
  TransientRBOperator(odeop′,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(
  rbsolver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  red_trial,
  red_test,
  (jac_lin,jac_nlin),
  (res_lin,res_nlin)
  )

  rbop_lin = reduced_operator(rbsolver,get_linear_operator(op),red_trial,red_test,jac_lin,res_lin)
  rbop_nlin = reduced_operator(rbsolver,get_nonlinear_operator(op),red_trial,red_test,jac_nlin,res_nlin)
  LinearNonlinearRBOperator(rbop_lin,rbop_nlin)
end

# function reduced_operator(
#   rbsolver::RBSolver,
#   op::ODEParamOperator{LinearNonlinearParamODE},
#   red_trial,
#   red_test,
#   (jac_lin,jac_nlin),
#   (res_lin,res_nlin)
#   )

#   rbop_lin = reduced_operator(rbsolver,get_linear_operator(op),red_trial,red_test,jac_lin,res_lin)
#   rbop_nlin = reduced_operator(rbsolver,get_nonlinear_operator(op),red_trial,red_test,jac_nlin,res_nlin)
#   LinearNonlinearTransientRBOperator(rbop_lin,rbop_nlin)
# end

function reduced_operator(rbsolver::RBSolver,feop::ParamOperator,sol::AbstractSnapshots,args...)
  op = get_algebraic_operator(feop)
  red_trial,red_test = reduced_spaces(rbsolver,feop,sol)
  reduced_operator(rbsolver,op,red_trial,red_test,args...)
end

function run_test(
  dir::String,rbsolver::RBSolver,feop::ParamOperator,tols=[1e-1,1e-2,1e-3,1e-4,1e-5],
  args...;nparams=10,kwargs...)

  fesnaps = try_loading_fe_snapshots(dir,rbsolver,feop,args...)

  μon = realization(feop;nparams,sampling=:uniform)
  x,festats = solution_snapshots(rbsolver,feop,μon,args...)

  perfs = ROMPerformance[]

  for tol in tols
    println("Running test $dir with tol = $tol")

    dir_tol = joinpath(dir,string(tol))
    create_dir(dir_tol)

    plot_dir_tol = joinpath(dir_tol,"plot")
    create_dir(plot_dir_tol)

    rbsolver = update_solver(rbsolver,tol)
    rbop = try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps)

    x̂,rbstats = solve(rbsolver,rbop,μon,args...)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
    println(perf)
    push!(perfs,perf)

    plot_a_solution(plot_dir_tol,feop,rbop,x,x̂,μon;kwargs...)
  end

  results_dir = joinpath(dir,"results")
  create_dir(results_dir)

  plot_errors(results_dir,tols,perfs)
  serialize(joinpath(results_dir,"performance.jld"),(tol => perf for (tol,perf) in zip(tols,perfs)))

  return perfs
end

export DrWatson
export Gridap
export Plots
export Serialization
export Test

export ROManifolds

export BlockMultiFieldStyle

export run_test

end
