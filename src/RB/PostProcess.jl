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

function ErrorTracker(id::Symbol,uh::Matrix{Float},uh_rb::Matrix{Float};Y=nothing)
  X = isnothing(Y) ? I(size(uh,1)) : Y
  relative_err,pointwise_err = compute_errors(uh,uh_rb,X)
  printstyled("Online relative error of variable $id is: $relative_err \n";
    color=:red)
  ErrorTracker(relative_err,pointwise_err)
end

function compute_errors(uh::Matrix{Float},uh_rb::Matrix{Float},X::AbstractMatrix)
  pointwise_err = abs.(uh-uh_rb)
  Nt = size(uh,2)
  absolute_err,uh_norm = zeros(Nt),zeros(Nt)
  for i = 1:Nt
    absolute_err[i] = sqrt((uh[:,i]-uh_rb[:,i])'*X*(uh[:,i]-uh_rb[:,i]))
    uh_norm[i] = sqrt(uh[:,i]'*X*uh[:,i])
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
  summary_path = joinpath(tests_path,"results_summary")
  create_dir!(summary_path)

  for tpath in get_all_subdirectories(tests_path)
    if tpath != summary_path
      case = last(split(tpath,'/'))
      ns,nt,toff = (),(),""
      tolpath = first(get_all_subdirectories(tpath))
      offpath = joinpath(tolpath,"offline")
      for varpath in get_all_subdirectories(offpath)
        var = last(split(varpath,'/'))
        bspath = joinpath(varpath,"basis_space")
        btpath = joinpath(varpath,"basis_time")
        if myisfile(bspath) && myisfile(btpath)
          ns = (ns...,(var,size(load(bspath),2)))
          nt = (nt...,(var,size(load(btpath),2)))
        end
      end
      onpath = joinpath(tolpath,"online")
      if myisfile(joinpath(onpath,"times"))
        toff = prod(load(joinpath(onpath,"times"))[1,2:3])
      end
      dict = merge(dict,Dict("ns_$(case)"=>ns,"nt_$(case)"=>nt,
        "toff_$(case)"=>toff))
    end
  end

  save(joinpath(summary_path,"offline_results"),dict)
end

function online_results_dict(info::RBInfo)
  dict = Dict("")
  tests_path = get_parent_dir(info.offline_path;nparent=3)
  summary_path = joinpath(tests_path,"results_summary")
  create_dir!(summary_path)

  for tpath in get_all_subdirectories(tests_path)
    if tpath != summary_path
      case = last(split(tpath,'/'))
      err_case_u,err_case_p,ton_case = NaN,NaN,NaN
      for tolpath in get_all_subdirectories(tpath)
        tol = parse(Float,last(split(tolpath,'/')))
        onpath = joinpath(tolpath,"online")
        if myisfile(joinpath(onpath,"errors_u"))
          err_case_u = last(load(joinpath(onpath,"errors_u")))
        end
        if myisfile(joinpath(onpath,"errors_p"))
          err_case_p = last(load(joinpath(onpath,"errors_p")))
        end
        if myisfile(joinpath(onpath,"times"))
          ton = load(joinpath(onpath,"times"))[2,2]
          ton_case = ton
        end
        dict = merge(dict,Dict("erru_$(case)_$tol"=>err_case_u,
        "errp_$(case)_$tol"=>err_case_p,"ton_$(case)_$tol"=>ton_case))
      end
    end
  end

  save(joinpath(summary_path,"online_results"),dict)
end

function H1_norm_matrix(opA::ParamBilinOperator,opM::ParamBilinOperator)
  afe = get_fe_function(opA)
  mfe = get_fe_function(opM)
  trial = realization_trial(opA)
  test = get_test(opA)
  A = assemble_matrix((u,v)->afe(1,u,v),trial,test)
  M = assemble_matrix((u,v)->mfe(1,u,v),trial,test)
  A+M
end

function L2_norm_matrix(opM::ParamBilinOperator)
  mfe = get_fe_function(opM)
  trial = realization_trial(opM)
  test = get_test(opM)
  assemble_matrix((u,v)->mfe(1,u,v),trial,test)
end
