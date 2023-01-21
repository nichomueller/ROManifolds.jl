mutable struct OfflineTime
  basis_time::Float
  assembly_time::Float
end

mutable struct TimeTracker
  offline_time::OfflineTime
  online_time::Float
end

struct ErrorTracker
  relative_err::Float
  pointwise_err::Matrix{Float}
end

function ErrorTracker(id::Symbol,uh::Matrix{Float},uh_rb::Matrix{Float},k::Int)
  relative_err,pointwise_err = compute_errors(uh,uh_rb)
  println("-----------------------------------------------------------------------------")
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

  println("----------------------------------------------------------------------------------")
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

function Gridap.writevtk(
  info::RBInfoSteady,
  s::Snapshots,
  X::FESpace,
  trian::Triangulation)

  id = get_id(s)
  path = joinpath(info.online_path,"$(id)h")
  fefun = FEFunction(X,s.snap[:,1])
  writevtk(trian,path,cellfields=["$(id)h"=>fefun])
end

function Gridap.writevtk(
  info::RBInfoSteady,
  res::RBResults,
  X::FESpace,
  trian::Triangulation)

  path = joinpath(info.online_path,"pwise_err_$(res.id)")
  fefun = FEFunction(X,res.pointwise_err[:,1])
  writevtk(trian,path,cellfields=["err"=>fefun])
end

function Gridap.writevtk(
  info::RBInfoUnsteady,
  tinfo::TimeInfo,
  s::Snapshots,
  X,
  trian::Triangulation)

  timesθ = get_timesθ(tinfo)
  id = get_id(s)
  path = joinpath(info.online_path,"$(id)h")

  createpvd(path) do pvd
    for (it,t) in enumerate(timesθ)
      fefun = FEFunction(X(t),s.snap[:,it])
      pvd = writevtk(trian,path*"_$(it).vtu",cellfields=["$(id)h"=>fefun])
    end
  end
end

function Gridap.writevtk(
  info::RBInfoUnsteady,
  tinfo::TimeInfo,
  res::RBResults,
  X,
  trian::Triangulation)

  timesθ = get_timesθ(tinfo)
  path = joinpath(info.online_path,"pwise_err_$(res.id)")

  createpvd(path) do pvd
    for (it,t) in enumerate(timesθ)
      fefun = FEFunction(X(t),res.pointwise_err[:,it])
      pvd = writevtk(trian,path*"_$(it).vtu",cellfields=["err"=>fefun])
    end
  end
end

struct MDEIMError
  id::Symbol
  offline_error::Float
  online_error::Float
end

function MDEIMError(
  id::NTuple{N,Symbol},
  offline_error::NTuple{N,Float},
  online_error::NTuple{N,Float}) where N

  MDEIMError.(id,offline_error,online_error)
end

function MDEIMError(
  op::RBVariable,
  mdeim,
  μ::Param,
  args...)

  id = get_id(mdeim)
  offline_error = mdeim_offline_error(op,mdeim,μ,args...)
  online_error = mdeim_online_error(op,mdeim,μ,args...)
  MDEIMError(id,offline_error,online_error)
end

#= function mdeim_offline_error(
  op::RBVariable,
  mdeim::MDEIM,
  μ::Param,)

  mat = evaluate(assemble_fe_structure(op),μ)
  mat_rb = rb_space_projection(op,mat)
  basis = get_basis(mdeim)
  infty_norm(mat-basis*basis'*mat)
end =#

function mdeim_online_error(
  op::RBVariable,
  mdeim,
  μ::Param,
  st_mdeim=false)

  mat = evaluate(assemble_fe_structure(op),μ)
  mat_rb = rb_projection(op,mat)
  mdeim_rb_tmp = online_assembler(op,mdeim,μ,st_mdeim)
  mdeim_rb = elim_shifted_matrix(mdeim_rb_tmp)
  infty_norm(mat_rb-mdeim_rb)
end

function infty_norm(q::AbstractArray)
  maximum(abs.(q))
end

function infty_norm(q::NTuple{N,AbstractArray}) where N
  infty_norm.(q)
end
