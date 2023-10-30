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

struct RBInfo
  ϵ::Float
  fe_path::String
  rb_path::String
  norm_style::Union{Symbol,Vector{Symbol}}
  compute_supremizers::Bool
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  st_mdeim::Bool
  postprocess::Bool
end

function RBInfo(test_path::String;ϵ=1e-4,norm_style=:l2,compute_supremizers=true,
  nsnaps_state=50,nsnaps_mdeim=20,nsnaps_test=10,st_mdeim=false,postprocess=false)

  fe_path = get_fe_path(test_path)
  rb_path = get_rb_path(test_path,ϵ;st_mdeim)
  RBInfo(ϵ,fe_path,rb_path,norm_style,compute_supremizers,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,st_mdeim,postprocess)
end

function get_norm_matrix(info::RBInfo,feop::PTFEOperator;norm_style=:l2)
  try
    T = get_vector_type(feop.test)
    load(info,SparseMatrixCSC{eltype(T),Int};norm_style)
  catch
    if norm_style == :l2
      nothing
    elseif norm_style == :L2
      get_L2_norm_matrix(feop)
    elseif norm_style == :H1
      get_H1_norm_matrix(feop)
    else
      @unreachable
    end
  end
end

function save(info::RBInfo,objs::Tuple)
  map(obj->save(info,obj),expand(objs))
end

function load(info::RBInfo,types::Tuple)
  map(type->load(info,type),expand(types))
end

function save(info::RBInfo,params::Table)
  path = joinpath(info.fe_path,"params")
  save(path,params)
end

function load(info::RBInfo,T::Type{Table})
  path = joinpath(info.fe_path,"params")
  load(path,T)
end

function save(info::RBInfo,norm_matrix::SparseMatrixCSC{T,Int};norm_style=:l2) where T
  path = joinpath(info.fe_path,"$(norm_style)_norm_matrix")
  save(path,norm_matrix)
end

function load(info::RBInfo,T::Type{SparseMatrixCSC{S,Int}};norm_style=:l2) where S
  path = joinpath(info.fe_path,"$(norm_style)_norm_matrix")
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
