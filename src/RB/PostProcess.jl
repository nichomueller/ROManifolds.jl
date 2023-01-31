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

function ErrorTracker(id::Symbol,uh::Matrix{Float},uh_rb::Matrix{Float})
  relative_err,pointwise_err = compute_errors(uh,uh_rb)
  printstyled("Online relative error of variable $id is: $relative_err \n";
    color=:red)
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

  relative_errs = Broadcasting(et->getproperty(et,:relative_err))(ets)
  pointwise_errs = Broadcasting(et->getproperty(et,:pointwise_err))(ets)
  et = ErrorTracker(sum(relative_errs)/nruns,sum(pointwise_errs)/nruns)
  ttnew = TimeTracker(tt.offline_time,tt.online_time/nruns)

  RBResults(id,ttnew,et)
end

function time_dict(r::RBResults)
  bt,at = r.tt.offline_time.basis_time,r.tt.offline_time.assembly_time
  Dict("offline_time"=>[bt,at],"online_time"=>r.tt.online_time)
end

err_dict(r::RBResults) = Dict("relative_err"=>r.et.relative_err,"pointwise_err"=>r.et.pointwise_err)

save(info::RBInfo,r::RBResults) = if info.save_online save(info.online_path,r) end

function save(path::String,r::RBResults)
  save(joinpath(path,"times"),time_dict(r))
  save(joinpath(path,"errors_$(r.id)"),err_dict(r))
end

function Gridap.writevtk(
  info::RBInfoSteady,
  s::Snapshots,
  X::FESpace,
  trian::Triangulation)

  id = get_id(s)
  plt_dir = joinpath(info.online_path,joinpath("plots"))
  create_dir!(plt_dir)

  path = joinpath(plt_dir,"$(id)h")
  fefun = FEFunction(X,s.snap[:,1])
  writevtk(trian,path,cellfields=["$(id)h"=>fefun])
end

function Gridap.writevtk(
  info::RBInfoSteady,
  res::RBResults,
  X::FESpace,
  trian::Triangulation)

  plt_dir = joinpath(info.online_path,joinpath("plots","pwise_err_$(res.id)"))
  create_dir!(plt_dir)

  fefun = FEFunction(X,res.et.pointwise_err[:,1])
  writevtk(trian,plt_dir,cellfields=["err"=>fefun])
end

function Gridap.writevtk(
  info::RBInfoUnsteady,
  tinfo::TimeInfo,
  s::Snapshots,
  X,
  trian::Triangulation)

  timesθ = get_timesθ(tinfo)
  id = get_id(s)
  plt_dir = joinpath(info.online_path,joinpath("plots"))
  create_dir!(plt_dir)

  path = joinpath(plt_dir,"$(id)h")
  for (it,t) in enumerate(timesθ)
    fefun = FEFunction(X(t),s.snap[:,it])
    writevtk(trian,path*"_$(it).vtu",cellfields=["$(id)h"=>fefun])
  end
end

function Gridap.writevtk(
  info::RBInfoUnsteady,
  tinfo::TimeInfo,
  res::RBResults,
  X,
  trian::Triangulation)

  timesθ = get_timesθ(tinfo)
  plt_dir = joinpath(info.online_path,joinpath("plots","pwise_err_$(res.id)"))
  create_dir!(plt_dir)

  for (it,t) in enumerate(timesθ)
    fefun = FEFunction(X(t),res.et.pointwise_err[:,it])
    writevtk(trian,plt_dir*"_$(it).vtu",cellfields=["err"=>fefun])
  end
end

function postprocess(info::RBInfo)
  offline_results_dict(info)
  online_results_dict(info)
end

function offline_results_dict(info::RBInfoUnsteady)
  dict = Dict("")
  tests_path = get_parent_dir(info.offline_path;nparent=3)

  for tpath in get_all_subdirectories(tests_path)
    case = tpath[findall(x->x=='/',tpath)[end]]
    tol_case,ns,nt,toff_case = Float[],Float[],Float[],Vector{Float}[]
    for tolpath in get_all_subdirectories(tpath)
      push!(tol_case,tolpath[findall(x->x=='/',tolpath)[end]])
      offpath = joinpath(tolpath,"offline")
      for varpath in get_all_subdirectories(offpath)
        push!(ns,size(load(joinpath(varpath,"basis_space")),2))
        push!(nt,size(load(joinpath(varpath,"basis_time")),2))
      end
      onpath = joinpath(tolpath,"online")
      if myisfile(joinpath(onpath,"times"))
        toff = load(joinpath(onpath,"times"))[:,1]
        push!(toff_case,toff)
      end
    end
    dict = merge(dict,Dict("tol_$case"=>tol_case,
      "err_$(case)_u"=>err_case_u,"err_$(case)_p"=>err_case_p,
      "toff_$(case)_case"=>toff_case,"ton_$(case)_case"=>ton_case))
  end

  save(joinpath(tests_path,"online_results"),dict)
end

function online_results_dict(info::RBInfo)
  dict = Dict("")
  tests_path = get_parent_dir(info.offline_path;nparent=3)

  for tpath in get_all_subdirectories(tests_path)
    case = tpath[findall(x->x=='/',dir)[end]]
    tol_case,err_case_u,err_case_p,ton_case = Float[],Float[],Float[],Float[]
    for tolpath in get_all_subdirectories(tpath)
      push!(tol_case,tolpath[findall(x->x=='/',dir)[end]])
      onpath = joinpath(tolpath,"online")
      if myisfile(joinpath(onpath,"errors_u"))
        push!(err_case_u,last(load(joinpath(onpath,"errors_u"))))
      end
      if myisfile(joinpath(onpath,"errors_p"))
        push!(err_case_p,last(load(joinpath(onpath,"errors_p"))))
      end
      if myisfile(joinpath(onpath,"times"))
        ton = last(load(joinpath(onpath,"times")))
        push!(ton_case,ton)
      end
    end
    dict = merge(dict,Dict("tol_$case"=>tol_case,"err_$(case)_u"=>err_case_u,
      "err_$(case)_p"=>err_case_p,"ton_$(case)_case"=>ton_case))
  end

  save(joinpath(tests_path,"online_results"),dict)
end

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
