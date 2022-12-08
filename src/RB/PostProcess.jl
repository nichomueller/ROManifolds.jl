mutable struct TimeTracker
  offline_time::Float
  online_time::Float
end

struct ErrorTracker
  err::Vector{Float}
  pointwise_err::Matrix{Float}
end

function ErrorTracker(id::Symbol,uh::Matrix{Float},uh_rb::Matrix{Float},k::Int)
  err,pointwise_err = compute_errors(uh,uh_rb)
  println("-----------------------------------------------------------")
  println("Online error of variable $id for μ=μ[$k] is: $(norm(err))")
  ErrorTracker(err,pointwise_err)
end

function compute_errors(uh::Matrix{Float},uh_rb::Matrix{Float})
  pointwise_err = abs.(uh-uh_rb)
  Nt = size(uh,2)
  err = zeros(Nt)
  for i = 1:Nt
    err[i] = norm(uh[:,i]-uh_rb[:,i])/norm(uh_rb[:,i])
  end
  err,pointwise_err
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

  errs = Broadcasting(et->getproperty(et,:err))(ets)
  pointwise_errs = Broadcasting(et->getproperty(et,:pointwise_err))(ets)
  et = ErrorTracker(sum(errs)/nruns,sum(pointwise_errs)/nruns)

  RBResults(id,tt,et)
end

time_dict(r::RBResults) = Dict("offline_time"=>r.tt.offline_time,"online_time"=>r.tt.online_time)
err_dict(r::RBResults) = Dict("err"=>r.et.err,"pointwise_err"=>r.et.pointwise_err)

save(info::RBInfo,r::RBResults) = if info.save_online save(info.online_path,r) end

function save(path::String,r::RBResults)
  save(joinpath(path,"times_$(r.id)"),time_dict(r))
  save(joinpath(path,"errors_$(r.id)"),err_dict(r))
end
