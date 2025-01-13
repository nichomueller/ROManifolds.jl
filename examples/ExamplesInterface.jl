module ExamplesInterface

using DrWatson
using Gridap
using Plots
using Serialization
using Test

using ROM

import Gridap.Helpers: @abstractmethod,@check
import Gridap.MultiField: BlockMultiFieldStyle
import ROM.RBSteady: solution_snapshots,get_state_reduction,get_residual_reduction,get_jacobian_reduction

function try_loading_reduced_operator(dir,rbsolver,feop,args...;kwargs...)
  try
    rbop = load_operator(dir,feop)
    println("Load reduced operator at $dir succeeded!")
    return rbop
  catch
    println("Load reduced operator at $dir failed, must run offline phase")
    rbop = reduced_operator(rbsolver,feop,args...;kwargs...)
    save(dir,rbop)
    return rbop
  end
end

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

get_error(perf::ROMPerformance) = perf.error

function plot_errors(dir,tols,perfs::Vector{ROMPerformance})
  errs = map(get_error,perfs)
  n = length(first(errs))
  errvec = map(i -> getindex.(errs,i),1:n)
  labvec = ["variable $i" for i in 1:n]

  plot_dir = joinpath(dir,"convergence.pdf")
  p = plot(tols,tols,lw=3,label="ϵ")
  scatter!(tols,errvec,lw=3,label=labvec)
  xlabel!("ϵ")
  ylabel!("Error")
  title!("Average relative error")

  savefig(p,plot_dir)
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

function run_test(dir::String,rbsolver::RBSolver,feop::ParamFEOperator,tols=[1e-1,1e-2,1e-3,1e-4,1e-5],args...)
  fesnaps = try_loading_fe_snapshots(dir,rbsolver,feop,args...)

  μon = realization(feop;nparams=10,random=true)
  x,festats = solution_snapshots(rbsolver,feop,μon,args...)

  perfs = ROMPerformance[]

  for tol in tols
    println("Running test $dir with tol = $tol")

    dir_tol = joinpath(dir,string(tol))
    create_dir(dir_tol)

    plot_dir_tol = joinpath(dir_tol,"plot")
    create_dir(plot_dir_tol)

    rbsolver = update_solver(rbsolver,tol)
    rbop = try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps,args...)

    x̂,rbstats = solve(rbsolver,rbop,μon,args...)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)
    push!(perfs,perf)

    plot_a_solution(plot_dir_tol,feop,rbop,x,x̂,μon)
  end

  results_dir = joinpath(dir,"results")
  create_dir(results_dir)

  plot_errors(results_dir,tols,perfs)
  serialize(joinpath(results_dir,"performance.jld"),(tol => perf for (tol,perf) in zip(tols,perfs)))

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
