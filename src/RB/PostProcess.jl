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

function ErrorTracker(
  uh::Snapshots,
  uh_rb::Snapshots;
  X=nothing)

  usnap = Matrix(get_snap(uh))
  usnap_rb = Matrix(get_snap(uh_rb))
  Y = isnothing(X) ? I(size(usnap,1)) : X
  relative_err,pointwise_err = compute_errors(usnap,usnap_rb,Y)

  ErrorTracker(relative_err,pointwise_err)
end

function compute_errors(
  uh::AbstractMatrix,
  uh_rb::AbstractMatrix,
  X::AbstractMatrix)

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
  nruns::Int
end

function RBResults(id::Symbol,tt::TimeTracker,ets::Vector{ErrorTracker})
  nruns = length(ets)

  relative_errs = Broadcasting(et->getproperty(et,:relative_err))(ets)
  pointwise_errs = Broadcasting(et->getproperty(et,:pointwise_err))(ets)
  et = ErrorTracker(sum(relative_errs)/nruns,sum(pointwise_errs)/nruns)
  ttnew = TimeTracker(tt.offline_time,tt.online_time/nruns)

  RBResults(id,ttnew,et,nruns)
end

function time_dict(r::RBResults)
  bt,at = r.tt.offline_time.basis_time,r.tt.offline_time.assembly_time
  Dict("offline_time"=>[bt,at],"online_time"=>r.tt.online_time)
end

err_dict(r::RBResults) = Dict("relative_err"=>r.et.relative_err,"pointwise_err"=>r.et.pointwise_err)

function save(info::RBInfo,r::RBResults)
  if info.save_online
    save(joinpath(info.online_path,"errors_$(r.id)"),err_dict(r))
    if !info.load_offline
      save(joinpath(info.online_path,"times"),time_dict(r))
    end
  end
  return nothing
end

function postprocess(
  info::RBInfo,
  res::NTuple{1,RBResults},
  fespaces::NTuple{1,FESpace},
  model::DiscreteModel,
  args...)

  res_u, = res
  rel_err_u = res_u.et.relative_err
  V, = fespaces

  printstyled("-------------------------------------------------------------\n")
  printstyled("Average online relative errors err_u: $rel_err_u\n";
    color=:red)
  printstyled("Average online wall time: $(tt.online_time/res_u.nruns) s\n";
    color=:red)
  printstyled("-------------------------------------------------------------\n")

  if info.save_online save(info,res) end

  if info.postprocess
    offline_results_dict(info)
    online_results_dict(info)
    trian = get_triangulation(model)
    writevtk(info,res_u,V,trian,args...)
  end

  return
end

function postprocess(
  info::RBInfo,
  res::NTuple{2,RBResults},
  fespaces::NTuple{2,FESpace},
  model::DiscreteModel,
  args...)

  res_u,res_p = res
  rel_err_u,rel_err_p = res_u.et.relative_err,res_p.et.relative_err
  V,Q = fespaces

  printstyled("-------------------------------------------------------------\n")
  printstyled("Average online relative errors (err_u,err_p): $((rel_err_u,rel_err_p))\n";
    color=:red)
  printstyled("Average online wall time: $(tt.online_time/res_u.nruns) s\n";
    color=:red)
  printstyled("-------------------------------------------------------------\n")

  if info.save_online save(info,res) end

  if info.postprocess
    offline_results_dict(info)
    online_results_dict(info)
    trian = get_triangulation(model)
    writevtk(info,res_u,V,trian,args...)
    writevtk(info,res_p,Q,trian,args...)
  end

  return
end

function Gridap.writevtk(
  info::RBInfoSteady,
  s::Snapshots,
  fespace::FESpace,
  trian::Triangulation)

  id = get_id(s)
  plt_dir = joinpath(info.online_path,joinpath("plots"))
  create_dir!(plt_dir)

  path = joinpath(plt_dir,"$(id)h")
  fefun = FEFunction(fespace,s.snap[:,1])
  writevtk(trian,path,cellfields=["$(id)h"=>fefun])
end

function Gridap.writevtk(
  info::RBInfoSteady,
  res::RBResults,
  fespace::FESpace,
  trian::Triangulation)

  plt_dir = joinpath(info.online_path,joinpath("plots","pwise_err_$(res.id)"))
  create_dir!(plt_dir)

  fefun = FEFunction(fespace,res.et.pointwise_err[:,1])
  writevtk(trian,plt_dir,cellfields=["err"=>fefun])
end

function Gridap.writevtk(
  info::RBInfoUnsteady,
  s::Snapshots,
  fespace::FESpace,
  trian::Triangulation,
  tinfo::TimeInfo)

  times = get_times(tinfo)
  id = get_id(s)
  plt_dir = joinpath(info.online_path,joinpath("plots"))
  create_dir!(plt_dir)

  path = joinpath(plt_dir,"$(id)h")
  for (it,t) in enumerate(times)
    fefun = FEFunction(fespace(t),Vector(s.snap[:,it]))
    writevtk(trian,path*"_$(it).vtu",cellfields=["$(id)h"=>fefun])
  end
end

function Gridap.writevtk(
  info::RBInfoUnsteady,
  res::RBResults,
  fespace::FESpace,
  trian::Triangulation,
  tinfo::TimeInfo)

  times = get_times(tinfo)
  plt_dir = joinpath(info.online_path,joinpath("plots","pwise_err_$(res.id)"))
  create_dir!(plt_dir)

  for (it,t) in enumerate(times)
    fefun = FEFunction(fespace(t),res.et.pointwise_err[:,it])
    writevtk(trian,plt_dir*"_$(it).vtu",cellfields=["err"=>fefun])
  end
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
  @assert get_id(opA) == :A && get_id(opM) == :M "wrong operators"
  assemble_affine_quantity(opA) + assemble_affine_quantity(opM)
end

function L2_norm_matrix(opM::ParamBilinOperator)
  @assert get_id(opM) == :M "wrong operator"
  assemble_affine_quantity(opM)
end
