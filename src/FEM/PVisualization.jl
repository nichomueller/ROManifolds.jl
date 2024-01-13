struct PVisualizationData{A<:AbstractArray}
  visdata::A
end

function Base.getproperty(x::PVisualizationData,sym::Symbol)
  if sym == :grid
    map(i->i.grid,x.visdata)
  elseif sym == :filebase
    r = nothing
    map(x.visdata) do visdata
      visdata.filebase
    end
    r
  elseif sym == :celldata
    map(i->i.celldata,x.visdata)
  elseif sym == :nodaldata
    map(i->i.nodaldata,x.visdata)
  else
    getfield(x,sym)
  end
end

struct PString{S<:AbstractVector{<:AbstractString}} <: AbstractString
  strings::S
end

function PString(filebase::String,n::Integer)
  filebases = map(1:n) do i
    joinpath(filebase,"param_$i")
  end
  PString(filebases)
end

Base.length(s::PString) = length(s.strings)
Base.iterate(s::PString) = iterate(s.strings)
Base.iterate(s::PString,i::Integer) = iterate(s.strings,i)

function Base.propertynames(x::PVisualizationData,private::Bool=false)
  (fieldnames(typeof(x))...,:grid,:filebase,:celldata,:nodaldata)
end

function Visualization.VisualizationData(
  grid::Grid,
  filebase::PString;
  celldata=Dict(),
  nodaldata=Dict())

  @assert isa(nodaldata,PArray)
  visdata = map(filebase,nodaldata) do filebase,nodaldata
    VisualizationData(grid,filebase;celldata,nodaldata)
  end
  PVisualizationData(visdata)
end

function Visualization._prepare_pdata(trian,cellfields::Dict{String,<:PCellField},samplingpoints)
  pdata = []
  for (k,v) in cellfields
    _v = _to_vector_cellfields(v)
    _pdata = map(_v) do _v
      _cellfields = Dict(k=>_v)
      Visualization._prepare_pdata(trian,_cellfields,samplingpoints)
    end
    push!(pdata,_pdata)
  end
  PArray(pdata)
end

function Visualization.write_vtk_file(
  grid::Grid,
  filebase::PString;
  celldata=Dict(),
  nodaldata=Dict())

  @assert isa(nodaldata,PArray)
  map(filebase,nodaldata) do filebase,nodaldata
    pvtk = Visualization.create_vtk_file(grid,filebase;celldata,nodaldata)
    map(vtk_save,pvtk)
  end
end
