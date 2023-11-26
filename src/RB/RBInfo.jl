function get_fe_path(tpath::String)
  create_dir(tpath)
  fepath = joinpath(tpath,"fem")
  create_dir(fepath)
  fepath
end

function get_rb_path(tpath::String,ϵ;st_mdeim=false)
  @assert isdir(tpath) "Provide valid path for the current test"
  keyword = st_mdeim ? "st" : "standard"
  outermost_path = joinpath(tpath,"rb")
  outer_path = joinpath(outermost_path,keyword)
  rb_path = joinpath(outer_path,"$ϵ")
  create_dir(rb_path)
  rb_path
end

struct RBInfo
  ϵ::Float
  fe_path::String
  rb_path::String
  norm_style::Symbol
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  st_mdeim::Bool
end

function RBInfo(
  test_path::String;
  ϵ=1e-4,
  norm_style=:l2,
  nsnaps_state=50,
  nsnaps_mdeim=20,
  nsnaps_test=10,
  st_mdeim=false)

  fe_path = get_fe_path(test_path)
  rb_path = get_rb_path(test_path,ϵ;st_mdeim)
  RBInfo(ϵ,fe_path,rb_path,norm_style,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,st_mdeim)
end

function get_norm_matrix(rbinfo::RBInfo,feop::PTFEOperator)
  norm_style = rbinfo.norm_style
  try
    T = get_vector_type(feop.test)
    load(rbinfo,SparseMatrixCSC{eltype(T),Int};norm_style)
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

struct BlockRBInfo
  ϵ::Float
  fe_path::String
  rb_path::String
  norm_style::Vector{Symbol}
  compute_supremizers::Bool
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  st_mdeim::Bool
end

function BlockRBInfo(
  test_path::String;
  ϵ=1e-4,
  norm_style=[:l2,:l2],
  compute_supremizers=true,
  nsnaps_state=50,
  nsnaps_mdeim=20,
  nsnaps_test=10,
  st_mdeim=false)

  fe_path = get_fe_path(test_path)
  rb_path = get_rb_path(test_path,ϵ;st_mdeim)
  BlockRBInfo(ϵ,fe_path,rb_path,norm_style,compute_supremizers,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,st_mdeim)
end

function Base.getindex(rbinfo::BlockRBInfo,i::Int)
  norm_style_i = rbinfo.norm_style[i]
  RBInfo(rbinfo.ϵ,rbinfo.fe_path,rbinfo.rb_path,norm_style_i,
    rbinfo.nsnaps_state,rbinfo.nsnaps_mdeim,rbinfo.nsnaps_test,rbinfo.st_mdeim)
end

function save(rbinfo,objs::Tuple,args...;kwargs...)
  map(obj->save(rbinfo,obj,args...;kwargs...),objs)
end

function load(rbinfo,types::Tuple,args...;kwargs...)
  map(type->load(rbinfo,type,args...;kwargs...),types)
end

function save(rbinfo,params::Table)
  path = joinpath(rbinfo.fe_path,"params")
  save(path,params)
end

function load(rbinfo,T::Type{Table})
  path = joinpath(rbinfo.fe_path,"params")
  load(path,T)
end

function save(rbinfo,norm_matrix::SparseMatrixCSC{T,Int};norm_style=:l2) where T
  path = joinpath(rbinfo.fe_path,"$(norm_style)_norm_matrix")
  save(path,norm_matrix)
end

function load(rbinfo,T::Type{SparseMatrixCSC{S,Int}};norm_style=:l2) where S
  path = joinpath(rbinfo.fe_path,"$(norm_style)_norm_matrix")
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

function save(rbinfo,cinfo::ComputationInfo)
  path = joinpath(rbinfo.fe_path,"stats")
  save(path,cinfo)
end

function load(rbinfo,T::Type{ComputationInfo})
  path = joinpath(rbinfo.fe_path,"stats")
  load(path,T)
end
