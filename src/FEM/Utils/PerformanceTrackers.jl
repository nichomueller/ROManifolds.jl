
abstract type PerformanceTracker end

mutable struct CostTracker <: PerformanceTracker
  time::Float64
  nallocs::Float64
  nruns::Int
end

function CostTracker(;time=0.0,nallocs=0.0,nruns=0)
  CostTracker(time,nallocs,nruns)
end

function CostTracker(stats::NamedTuple)
  time = stats[:time]
  nallocs = stats[:bytes] / 1e6
  nruns = 1
  CostTracker(time,nallocs,nruns)
end

function Base.show(io::IO,k::MIME"text/plain",t::CostTracker)
  println(io," > computational time (s) across $(t.nruns) runs: $(t.time)")
  println(io," > memory footprint (Mb) across $(t.nruns) runs: $(t.nallocs)")
end

function Base.copyto!(t1::CostTracker,t2::CostTracker)
  t1.time = t2.time
  t1.nallocs = t2.nallocs
  t1.nruns = t2.nruns
end

function reset_tracker!(t::CostTracker)
  t.time = 0.0
  t.nallocs = 0.0
  t.nruns = 0
end

function update_tracker!(t::CostTracker,stats::NamedTuple,nruns=t.nruns+1;msg="")
  time = stats[:time]
  nallocs = stats[:bytes] / 1e6
  t.time += time
  t.nallocs += nallocs
  t.nruns = nruns
  if !isempty(msg)
    println(msg)
    show(stdout,MIME"text/plain"(),t)
  end
end

function get_stats(t::CostTracker;verbose=true)
  avg_time = t.time / t.nruns
  avg_nallocs = t.nallocs / t.nruns
  if verbose
    println("Average time (s): $avg_time")
    println("Average number of allocations (Mb): $avg_nallocs")
  end
  return avg_time,avg_nallocs
end

"""
    compute_speedup(t1::CostTracker,t2::CostTracker) -> (Float64,Float64)

Computes the speedup the tracker `t2` achieves with respect to `t1`, in time and
in memory footprint
"""
function compute_speedup(t1::CostTracker,t2::CostTracker;verbose=true)
  avg_time1,avg_nallocs1 = get_stats(t1;verbose=false)
  avg_time2,avg_nallocs2 = get_stats(t2;verbose=false)
  speedup_time = avg_time1 / avg_time2
  speedup_memory = avg_nallocs1 / avg_nallocs2
  if verbose
    println("Speedup in time: $(speedup_time)")
    println("Speedup in memory: $(speedup_memory)")
  end
  return speedup_time,speedup_memory
end

mutable struct GenericPerformance <: PerformanceTracker
  error::Vector{Float64}
  cost::CostTracker
end

function Base.show(io::IO,k::MIME"text/plain",p::GenericPerformance)
  show(io,MIME"text/plain"(),p.cost)
  println(io," > errors across $(t.nruns) runs: $(t.error)")
end

function Base.copyto!(p1::GenericPerformance,p2::GenericPerformance)
  copyto!(p1.error,p2.error)
  copyto!(p1.cost,p2.cost)
end

function reset_tracker!(p::GenericPerformance)
  reset_tracker!(p.cost)
end

function update_tracker!(p::GenericPerformance,stats::NamedTuple,args...;kwargs...)
  update_tracker!(p.cost,stats,args...;kwargs...)
end

function get_stats(p::GenericPerformance;verbose=true)
  avg_time,avg_nallocs = get_stats(p.cost;verbose)
  avg_error = mean(t.error)
  if verbose
    println("Average error: $avg_error")
  end
  return avg_time,avg_nallocs,avg_error
end

function compute_speedup(p1::GenericPerformance,p2::GenericPerformance;kwargs...)
  compute_speedup(p1.cost,p2.cost;kwargs...)
end

induced_norm(v::AbstractVector,args...) = norm(v)
induced_norm(v::AbstractVector,norm_matrix::AbstractMatrix) = sqrt(v'*norm_matrix*v)

function compute_error(
  sol::AbstractArray{T,N},
  sol_approx::AbstractArray{T,N},
  args...;nruns=size(sol,N)
  ) where {T,N}

  @check size(sol) == size(sol_approx)
  errors = zeros(nruns)
  @inbounds for i = 1:nruns
    soli = selectdim(sol,N,i)
    soli_approx = selectdim(sol_approx,N,i)
    errors[i] = compute_relative_error(soli,soli_approx,args...)
  end
  return errors
end

function compute_relative_error(sol::AbstractArray,sol_approx::AbstractArray,args...)
  err_norm = induced_norm(soli-soli_approx,args...)
  sol_norm = induced_norm(soli,args...)
  rel_norm = err_norm / sol_norm
  return rel_norm
end
