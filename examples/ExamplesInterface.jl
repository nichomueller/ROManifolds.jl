module ExamplesInterface

using DrWatson
using Gridap
using Plots
using Serialization
using Test

using ROM

import Gridap.Helpers: @abstractmethod
import Gridap.MultiField: BlockMultiFieldStyle
import ROM.RBSteady: solution_snapshots

function try_loading_reduced_operator(dir,rbsolver,feop,args...;kwargs...)
  try
    rbop = load_operator(dir,feop)
    println("Load succeeded!")
  catch
    println("Load failed, must run offline phase")
    rbop = reduced_operator(rbsolver,feop,args...;kwargs...)
    save(dir,rbop)
  end
  return rbop
end

function try_loading_fe_snapshots(dir,rbsolver,feop,args...;kwargs...)
  try
    fesnaps = load_snapshots(dir)
    println("Load succeeded!")
  catch
    println("Load failed, must compute snapshots")
    fesnaps, = solution_snapshots(rbsolver,feop,args...;kwargs...)
    save(fesnaps,dir)
  end
  return fesnaps
end

function plot_errors(dir,tols,perfs)
  plot_dir = joinpath(dir,"plot.pdf")
  p = plot(tols,perfs,lw=3)
  xlabel!("ϵ")
  ylabel!("Error")
  title!("Average relative error")
  save(plot_dir,p)
end

update_redstyle(::SearchSVDRank,tol) = SearchSVDRank(tol)
update_redstyle(::LRApproxRank,tol) = LRApproxRank(tol)
update_redstyle(::TTSVDRanks,tol) = TTSVDRanks(tol)

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
    update_reduction(red.reduction_space),
    update_reduction(red.reduction_time,tol)
    )
end

function update_reduction(red::TransientMDEIMReduction,tol)
  TransientMDEIMReduction(update_reduction(red.reduction),red.combine)
end

function update_solver(rbsolver::RBSolver,tol)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = update_reduction(get_state_reduction(rbsolver),tol)
  residual_reduction = update_reduction(get_residual_reduction(rbsolver),tol)
  jacobian_reduction = update_reduction(get_jacobian_reduction(rbsolver),tol)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function run_test(dir::String,rbsolver::RBSolver,feop::ParamFEOperator,tols=[1e-1,1e-2,1e-3,1e-4,1e-5])
  fesnaps = try_loading_fe_snapshots(dir,rbsolver,feop)

  μon = realization(feop;nparams=10,random=true)
  x,festats = solution_snapshots(rbsolver,feop,μon)

  perfs = ROMPerformance[]

  for tol in tols
    dir_tol = joinpath(dir,string(tol))
    create_dir(dir_tol)

    rbsolver = update_solver(rbsolver,tol)
    rbop = try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps)

    x,festats = solve(rbsolver,feop,μon)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)
    push!(perfs,perf)
  end

  plot_errors(tols,perfs)
  save(dir,(tol => perf for (tol,perf) in zip(tols,perfs)))

end

export DrWatson
export Gridap
export Plots
export Serialization
export Test

export ROM

export BlockMultiFieldStyle

export run_test

end
