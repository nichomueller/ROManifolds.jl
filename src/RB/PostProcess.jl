struct OfflineTime
  basis_time::Float
  assembly_time::Float
end

function save(info::RBInfo,ot::OfflineTime)
  ot_dict = Dict("basis_time"=>ot.basis_time,"assembly_time"=>ot.assembly_time)
  save(joinpath(info.offline_path,"offline_time"),ot_dict)
end

struct RBResults
  id::Symbol
  relative_err::Float
  pointwise_err::Matrix{Float}
  online_time::Float
end

function RBResults(
  sol::NTuple{N,Snapshots},
  sol_approx::NTuple{N,Snapshots},
  args...;
  kwargs...)::NTuple{N,RBResults} where N

  RBResults.(sol,sol_approx;kwargs...)
end

function RBResults(
  sol::Snapshots,
  sol_approx::Snapshots,
  online_time::Float;
  X=nothing)

  id = get_id(sol)
  sol_mat = get_snap(Matrix{Float},sol)
  sol_approx_mat = get_snap(Matrix{Float},sol_approx)
  Y = isnothing(X) ? I(size(sol_mat,1)) : X
  relative_err,pointwise_err = compute_errors(sol_mat,sol_approx_mat,Y)

  RBResults(id,relative_err,pointwise_err,online_time)
end

function RBResults(res::Vector{RBResults})
  nruns = length(res)

  id = first(res).id
  relative_errs = Broadcasting(r->getproperty(r,:relative_err))(res)
  online_times = Broadcasting(r->getproperty(r,:online_time))(res)

  relative_err = sum(relative_errs)/nruns
  pointwise_err = first(res).pointwise_err
  online_time = sum(online_times/nruns)

  RBResults(id,relative_err,pointwise_err,online_time)
end

function save(info::RBInfo,res::NTuple{1,RBResults})
  res_u, = res
  res_dict = Dict("relative_err_$(res_u.id)" => res_u.relative_err,
                  "online_time"=>res_u.online_time)
  save(joinpath(info.online_path,"results"),res_dict)
end

function save(info::RBInfo,res::NTuple{2,RBResults})
  res_u,res_p = res
  res_dict = Dict("relative_err_$(res_u.id)" => res_u.relative_err,
                  "relative_err_$(res_p.id)" => res_p.relative_err,
                  "online_time"=>res_u.online_time)
  save(joinpath(info.online_path,"results_$id"),res_dict)
end

function compute_errors(
  u::AbstractMatrix,
  uh_rb::AbstractMatrix,
  X::AbstractMatrix)

  pointwise_err = abs.(u-uh_rb)
  Nt = size(u,2)
  absolute_err,uh_norm = zeros(Nt),zeros(Nt)
  for i = 1:Nt
    absolute_err[i] = sqrt((u[:,i]-uh_rb[:,i])'*X*(u[:,i]-uh_rb[:,i]))
    uh_norm[i] = sqrt(u[:,i]'*X*u[:,i])
  end
  relative_err = norm(absolute_err)/norm(uh_norm)

  relative_err,pointwise_err
end

function postprocess(
  info::RBInfo,
  res::NTuple{1,RBResults},
  fespaces::NTuple{1,FESpace},
  model::DiscreteModel,
  args...)

  res_u, = res
  V, = fespaces

  printstyled("-------------------------------------------------------------\n")
  printstyled("Average online relative errors err_u: $(res_u.relative_err)\n";
    color=:red)
  printstyled("Average online wall time: $(res_u.online_time) s\n";
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
  rel_err_u,rel_err_p = res_u.relative_err,res_p.relative_err
  V,Q = fespaces

  printstyled("-------------------------------------------------------------\n")
  printstyled("Average online relative errors (err_u,err_p): $((rel_err_u,rel_err_p))\n";
    color=:red)
  printstyled("Average online wall time: $(res_u.online_time) s\n";
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
