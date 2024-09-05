
abstract type PerformanceTracker end

mutable struct CostTracker <: PerformanceTracker
  time::Float64
  nallocs::Float64
  nruns::Int
end

function CostTracker(;time=0.0,nallocs=0.0,nruns=0)
  CostTracker(time,nallocs,nruns)
end

function CostTracker(stats::NamedTuple,nruns=1)
  time = stats[:time]
  nallocs = stats[:bytes] / 1e6
  CostTracker(time,nallocs,nruns)
end

function Base.show(io::IO,k::MIME"text/plain",t::CostTracker)
  println(io," ---------------------- CostTracker ----------------------------")
  println(io," > computational time (s) across $(t.nruns) runs: $(t.time)")
  println(io," > memory footprint (Mb) across $(t.nruns) runs: $(t.nallocs)")
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,t::CostTracker)
  show(io,MIME"text/plain"(),t)
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

function get_stats(t::CostTracker)
  avg_time = t.time / t.nruns
  avg_nallocs = t.nallocs / t.nruns
  return avg_time,avg_nallocs
end

struct SU <: PerformanceTracker
  speedup_time::Float64
  speedup_memory::Float64
end

function Base.show(io::IO,k::MIME"text/plain",su::SU)
  println(io," -------------------------- SU -------------------------------")
  println(io," > speedup in time: $(su.speedup_time)")
  println(io," > speedup in memory: $(su.speedup_memory)")
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,su::SU)
  show(io,MIME"text/plain"(),su)
end

"""
    compute_speedup(t1::CostTracker,t2::CostTracker) -> SU

Computes the speedup the tracker `t2` achieves with respect to `t1`, in time and
in memory footprint
"""
function compute_speedup(t1::CostTracker,t2::CostTracker)
  avg_time1,avg_nallocs1 = get_stats(t1)
  avg_time2,avg_nallocs2 = get_stats(t2)
  speedup_time = avg_time1 / avg_time2
  speedup_memory = avg_nallocs1 / avg_nallocs2
  return SU(speedup_time,speedup_memory)
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
  return mean(errors)
end

function compute_relative_error(sol::AbstractArray,sol_approx::AbstractArray,args...)
  err_norm = induced_norm(sol-sol_approx,args...)
  sol_norm = induced_norm(sol,args...)
  rel_norm = err_norm / sol_norm
  return rel_norm
end
