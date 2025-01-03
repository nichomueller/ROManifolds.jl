
abstract type PerformanceTracker end

mutable struct CostTracker <: PerformanceTracker
  name::String
  time::Float64
  nallocs::Float64
  nruns::Int
end

function CostTracker(;name="",time=0.0,nallocs=0.0,nruns=1)
  CostTracker(name,time,nallocs,nruns)
end

function CostTracker(stats::NamedTuple;nruns=1,name="")
  time = stats[:time]
  nallocs = stats[:bytes] / 1e6
  CostTracker(name,time,nallocs,nruns)
end

function Base.show(io::IO,k::MIME"text/plain",t::CostTracker)
  show_mega = t.nallocs < 1e3
  println(io," -------------------------------------------------------------")
  println(io," > CostTracker($(t.name)) across $(t.nruns) runs:")
  println(io," > computational time (s): $(t.time)")
  if show_mega
    println(io," > memory footprint (Mb): $(t.nallocs)")
  else
    println(io," > memory footprint (Gb): $(t.nallocs/1e3)")
  end
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,t::CostTracker)
  show(io,MIME"text/plain"(),t)
end

function reset_tracker!(t::CostTracker)
  t.time = 0.0
  t.nallocs = 0.0
  t.nruns = 0
end

function update_tracker!(t::CostTracker,stats::NamedTuple;msg="")
  time = stats[:time]
  nallocs = stats[:bytes] / 1e6
  t.time += time
  t.nallocs += nallocs
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
  name::String
  speedup_time::Float64
  speedup_memory::Float64
end

function Base.show(io::IO,k::MIME"text/plain",su::SU)
  println(io," -------------------- SU($(su.name)) -------------------------")
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
  name = "$(t1.name) / $(t2.name)"
  avg_time1,avg_nallocs1 = get_stats(t1)
  avg_time2,avg_nallocs2 = get_stats(t2)
  speedup_time = avg_time1 / avg_time2
  speedup_memory = avg_nallocs1 / avg_nallocs2
  return SU(name,speedup_time,speedup_memory)
end

induced_norm(v::AbstractVector) = norm(v)
induced_norm(v::AbstractVector,norm_matrix::AbstractMatrix) = sqrt(v'*norm_matrix*v)

induced_norm(A::AbstractMatrix) = sqrt.(diag(A'*A))
induced_norm(A::AbstractMatrix,norm_matrix::AbstractMatrix) = sqrt.(diag(A'*norm_matrix*A))

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
