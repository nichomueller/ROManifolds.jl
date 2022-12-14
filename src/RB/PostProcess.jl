mutable struct TimeTracker
  offline_time::Float
  online_time::Float
end

struct ErrorTracker
  relative_err::Float
  pointwise_err::Matrix{Float}
end

function ErrorTracker(id::Symbol,uh::Matrix{Float},uh_rb::Matrix{Float},k::Int)
  relative_err,pointwise_err = compute_errors(uh,uh_rb)
  println("-----------------------------------------------------------")
  println("Online relative error of variable $id for μ=μ[$k] is: $relative_err")
  ErrorTracker(relative_err,pointwise_err)
end

function compute_errors(uh::Matrix{Float},uh_rb::Matrix{Float})
  pointwise_err = abs.(uh-uh_rb)
  Nt = size(uh,2)
  absolute_err,uh_norm = zeros(Nt),zeros(Nt)
  for i = 1:Nt
    absolute_err[i] = norm(uh[:,i]-uh_rb[:,i])
    uh_norm[i] = norm(uh[:,i])
  end
  relative_err = norm(absolute_err)/norm(uh_norm)
  relative_err,pointwise_err
end

mutable struct RBResults
  id::Symbol
  tt::TimeTracker
  et::ErrorTracker
end

function RBResults(id::Symbol,tt::TimeTracker,ets::Vector{ErrorTracker})
  nruns = length(ets)

  println("----------------------------------------------------------------")
  println("Average online wall time: $(tt.online_time/nruns) s")

  relative_errs = Broadcasting(et->getproperty(et,:relative_err))(ets)
  pointwise_errs = Broadcasting(et->getproperty(et,:pointwise_err))(ets)
  et = ErrorTracker(sum(relative_errs)/nruns,sum(pointwise_errs)/nruns)

  RBResults(id,tt,et)
end

time_dict(r::RBResults) = Dict("offline_time"=>r.tt.offline_time,"online_time"=>r.tt.online_time)
err_dict(r::RBResults) = Dict("relative_err"=>r.et.relative_err,"pointwise_err"=>r.et.pointwise_err)

save(info::RBInfo,r::RBResults) = if info.save_online save(info.online_path,r) end

function save(path::String,r::RBResults)
  save(joinpath(path,"times_$(r.id)"),time_dict(r))
  save(joinpath(path,"errors_$(r.id)"),err_dict(r))
end
