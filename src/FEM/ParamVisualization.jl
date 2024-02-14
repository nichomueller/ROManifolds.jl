"""Recursive creation of a directory"""
function create_dir(dir::String)
  if !isdir(dir)
    parent_dir, = splitdir(dir)
    create_dir(parent_dir)
    mkdir(dir)
  end
  return
end

# function get_parent_dir(dir::String;nparent=1)
#   dir = dir[1:findall(x->x=='/',dir)[end]-1]
#   for _ = 1:nparent-1
#     dir = dir[1:findall(x->x=='/',dir)[end]-1]
#   end
#   dir
# end

"""Get a full list of subdirectories at a given root directory"""
function get_all_subdirectories(path::String)
  filter(isdir,readdir(path,join=true))
end

struct ParamString{S} <: AbstractString
  strings::S
end

function ParamString(dir::String,r::ParamRealization)
  np = num_params(r)
  strings = map(1:np) do i
    file = joinpath(dir,"param_$i")
    create_dir(file)
    file
  end
  ParamString(strings)
end

function ParamString(dir::String,r::FEM.TransientParamRealizationAt)
  np = num_params(r)
  t = get_times(r)
  strings = map(1:np) do i
    f = joinpath(dir,"param_$i")
    create_dir(f)
    joinpath(f,"solution_$t"*".vtu")
  end
  ParamString(strings)
end

Base.length(s::ParamString) = length(s.strings)
Base.iterate(s::ParamString,i::Integer...) = iterate(s.strings,i...)

function Base.show(io::IO,::MIME"text/plain",s::ParamString)
  for string in s.strings
    show(string); print("\n")
  end
end

struct ParamVisualizationData
  grid::Grid
  filebase::ParamString
  celldata
  nodaldata
  function ParamVisualizationData(
    grid::Grid,
    filebase::ParamString;
    celldata=Dict(),
    nodaldata=ParamContainer([Dict()]))
    new(grid,filebase,celldata,nodaldata)
  end
end

function Visualization.VisualizationData(
  grid::Grid,
  filebase::ParamString;
  celldata=Dict(),
  nodaldata=ParamContainer([Dict()]))

  ParamVisualizationData(grid,filebase;celldata,nodaldata)
end

function Visualization.write_vtk_file(
  trian::Grid,
  filebase::ParamString;
  celldata=Dict(),
  nodaldata=ParamContainer([Dict()]))

  @assert isa(nodaldata,ParamContainer)
  map(filebase,nodaldata) do filebase,nodaldata
    pvtk = Visualization.create_vtk_file(trian,filebase;celldata,nodaldata)
    Visualization.vtk_save(pvtk)
  end
end

function Visualization.create_vtk_file(
  trian::Grid,
  filebase::ParamString;
  nodaldata=ParamContainer([Dict()]),
  kwargs...)

  @assert isa(nodaldata,ParamContainer)
  map(filebase,nodaldata) do filebase,nodaldata
    Visualization.create_vtk_file(trian,filebase;nodaldata,kwargs...)
  end
end

function Visualization.create_pvtk_file(
  trian::Grid,
  filebase::ParamString;
  nodaldata=ParamContainer([Dict()]),
  kwargs...)

  @assert isa(nodaldata,ParamContainer)
  map(filebase,nodaldata) do filebase,nodaldata
    Visualization.create_pvtk_file(trian,filebase;nodaldata,kwargs...)
  end
end

function Visualization.createpvd(
  r::TransientParamRealizationAt,
  args...;kwargs...)

  map(r) do ri
    WriteVTK.paraview_collection(args...;kwargs...)
  end
end

function Visualization.createpvd(
  parts::Union{Nothing,AbstractArray},
  r::TransientParamRealizationAt,
  args...;kwargs...)

  map(r) do ri
    Visualization.createpvd(parts,args...;kwargs...)
  end
end

function Visualization.createpvd(
  f,
  r::TransientParamRealizationAt,
  args...;kwargs...)

  pvd = Visualization.createpvd(r,args...;kwargs...)
  try
    f(pvd)
  finally
    Visualization.savepvd(pvd)
  end
end

function Visualization.createpvd(
  f,
  parts::Union{Nothing,AbstractArray},
  r::TransientParamRealizationAt,
  args...;kwargs...)

  pvd = Visualization.createpvd(parts,r,args...;kwargs...)
  try
    f(pvd)
  finally
    Visualization.savepvd(pvd)
  end
end

function Visualization.savepvd(pvd::Vector{<:WriteVTK.CollectionFile})
  map(pvd) do pvd
    Visualization.vtk_save(pvd)
  end
end

function Base.setindex!(
  pvd::Vector{<:WriteVTK.CollectionFile},
  pvtk::AbstractArray,
  r::TransientParamRealizationAt)

  time = get_times(r)
  map(vtk_save,pvtk)
  map(pvtk,pvd) do pvtk,pvd
    pvd[time] = pvtk
  end
end

function _lengths(cellfields::Vector{<:Pair{String,<:ParamFEFunction}})
  lengths = []
  for (k,v) in cellfields
    push!(lengths,length(v))
  end
  L = lengths[1]
  @assert all(lengths .== L)
  return L
end

function Visualization._prepare_pdata(trian,cellfields::Vector{<:Pair{String,<:ParamFEFunction}},samplingpoints)
  L = _lengths(cellfields)
  x = CellPoint(samplingpoints,trian,ReferenceDomain())
  pdata = map(1:L) do i
    pdatai = Dict()
    for (k,v) in cellfields
      _vi = CellField(_getindex(v,i),trian)
      pdatai[k], = Visualization._prepare_node_to_coords(evaluate(_vi,x))
    end
    pdatai
  end
  ParamContainer(pdata)
end
