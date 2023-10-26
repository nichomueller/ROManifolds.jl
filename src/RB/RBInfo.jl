struct RBInfo
  ϵ::Float
  fe_path::String
  rb_path::String
  energy_norm::Union{Symbol,Vector{Symbol}}
  compute_supremizers::Bool
  st_mdeim::Bool
  postprocess::Bool
end

function RBInfo(test_path::String;ϵ=1e-4,energy_norm=:l2,
  compute_supremizers=true,st_mdeim=false,postprocess=false)

  fe_path = get_fe_path(test_path)
  rb_path = get_rb_path(test_path,ϵ;st_mdeim)
  RBInfo(ϵ,fe_path,rb_path,energy_norm,compute_supremizers,st_mdeim,postprocess)
end

function get_fe_path(tpath::String)
  create_dir!(tpath)
  fepath = joinpath(tpath,"fem")
  create_dir!(fepath)
  fepath
end

function get_rb_path(tpath::String,ϵ::Float;st_mdeim=false)
  @assert isdir(tpath) "Provide valid path for the current test"
  keyword = st_mdeim ? "st" : "standard"
  outermost_path = joinpath(tpath,"rb")
  outer_path = joinpath(outermost_path,keyword)
  rb_path = joinpath(outer_path,"$ϵ")
  create_dir!(rb_path)
  rb_path
end

for (fsave,fload) in zip((:save,:save_test),(:load,:load_test))
  @eval begin
    function $fsave(info::RBInfo,objs::Tuple)
      map(obj->$fsave(info,obj),expand(objs))
    end

    function $fload(info::RBInfo,types::Tuple)
      map(type->$fload(info,type),expand(types))
    end
  end
end

function save(info::RBInfo,params::Table)
  path = joinpath(info.fe_path,"params")
  save(path,params)
end

function load(info::RBInfo,T::Type{Table})
  path = joinpath(info.fe_path,"params")
  load(path,T)
end

struct ComputationInfo
  avg_time::Float
  avg_nallocs::Float
  function ComputationInfo(stats::NamedTuple,nruns::Int)
    avg_time = stats[:time] / nruns
    avg_nallocs = stats[:bytes] / (1e6*nruns)
    new(avg_time,avg_nallocs)
  end
end

get_avg_time(cinfo::ComputationInfo) = cinfo.avg_time
get_avg_nallocs(cinfo::ComputationInfo) = cinfo.avg_nallocs

function save(info::RBInfo,cinfo::ComputationInfo)
  path = joinpath(info.fe_path,"stats")
  save(path,cinfo)
end

function load(info::RBInfo,T::Type{ComputationInfo})
  path = joinpath(info.fe_path,"stats")
  load(path,T)
end
