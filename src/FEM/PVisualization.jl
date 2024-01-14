struct PString{S<:AbstractVector{<:AbstractString}}
  strings::S
end

function PString(filebase::String,n::Integer)
  filebases = map(1:n) do i
    f = joinpath(filebase,"param_$i")
    create_dir(f)
    f
  end
  PString(filebases)
end

Base.length(s::PString) = length(s.strings)
Base.iterate(s::PString) = iterate(s.strings)
Base.iterate(s::PString,i::Integer) = iterate(s.strings,i)

function Base.show(io::IO,::MIME"text/plain",s::PString)
  for string in s.strings
    show(string); print("\n")
  end
end

struct PVisualizationData{A<:AbstractArray}
  visdata::A
end

function PVisualizationData(
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

function Base.getproperty(x::PVisualizationData,sym::Symbol)
  if sym == :grid
    x.visdata[1].grid
  elseif sym == :filebase
    map(i->i.filebase,x.visdata)
  elseif sym == :celldata
    x.visdata[1].celldata
  elseif sym == :nodaldata
    map(i->i.nodaldata,x.visdata)
  else
    getfield(x,sym)
  end
end

function Base.propertynames(x::PVisualizationData,private::Bool=false)
  (fieldnames(typeof(x))...,:grid,:filebase,:celldata,:nodaldata)
end

function Visualization.visualization_data(
  grid::Grid,
  filebase::PString;
  celldata=Dict(),nodaldata=Dict())
  (PVisualizationData(grid,filebase;celldata=celldata,nodaldata=nodaldata),)
end

function Visualization.visualization_data(
  trian::Triangulation,
  filebase::PString;
  order=-1,nsubcells=-1,celldata=Dict(),cellfields=Dict())

  if order == -1 && nsubcells == -1
    f = (reffe) -> UnstructuredGrid(reffe)
  elseif order != -1 && nsubcells == -1
    f = (reffe) -> UnstructuredGrid(LagrangianRefFE(Float64,get_polytope(reffe),order))
  elseif order == -1 && nsubcells != -1
    f = (reffe) -> UnstructuredGrid(compute_reference_grid(reffe,nsubcells))
  else
    @unreachable "order and nsubcells kw-arguments can not be given at the same time"
  end

  ref_grids = map(f,get_reffes(trian))
  visgrid = Visualization.VisualizationGrid(trian,ref_grids)

  cdata = Visualization._prepare_cdata(celldata,visgrid.sub_cell_to_cell)
  pdata = Visualization._prepare_pdata(trian,cellfields,visgrid.cell_to_refpoints)

  (PVisualizationData(visgrid,filebase;celldata=cdata,nodaldata=pdata),)
end

function _lengths(cellfields::Dict{String,<:PCellField})
  lengths = []
  for (k,v) in cellfields
    push!(lengths,length(v))
  end
  L = lengths[1]
  @assert all(lengths .== L)
  return L
end

function Visualization._prepare_pdata(trian,cellfields::Dict{String,<:PCellField},samplingpoints)
  L = _lengths(cellfields)
  x = CellPoint(samplingpoints,trian,ReferenceDomain())
  pdata = map(1:L) do i
    pdatai = Dict()
    for (k,v) in cellfields
      _vi = CellField(v[i],trian)
      pdatai[k], = Visualization._prepare_node_to_coords(evaluate(_vi,x))
    end
    pdatai
  end
  PArray(pdata)
end

function Visualization.write_vtk_file(
  trian::Grid,
  filebase::AbstractVector;
  celldata=Dict(),
  nodaldata=Dict())

  @assert isa(nodaldata,AbstractVector)
  map(filebase,nodaldata) do filebase,nodaldata
    pvtk = Visualization.create_vtk_file(trian,filebase;celldata,nodaldata)
    Visualization.vtk_save(pvtk)
  end
end
