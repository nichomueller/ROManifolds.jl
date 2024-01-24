function _getindex(f::GenericCellField,index)
  data = get_data(f)
  trian = get_triangulation(f)
  DS = DomainStyle(f)
  di = getindex.(data,index)
  GenericCellField(di,trian,DS)
end

function _getindex(f::TrialPFESpace,index)
  dv = f.dirichlet_values[index]
  TrialFESpace(dv,f.space)
end

function _getindex(f::FESpaceToPFESpace,index)
  f.space
end

function _length(f::SingleFieldFEFunction)
  @assert length_dirichlet_values(f.fe_space) == length(f.dirichlet_values)
  length(f.dirichlet_values)
end

function _getindex(f::SingleFieldFEFunction,index)
  cf = _getindex(f.cell_field,index)
  fs = _getindex(f.fe_space,index)
  cv = f.cell_dof_values[index]
  fv = f.free_values[index]
  dv = f.dirichlet_values[index]
  SingleFieldFEFunction(cf,cv,fv,dv,fs)
end

_length(f::MultiFieldFEFunction) = _length(first(f.single_fe_functions))

function _getindex(f::MultiFieldFEFunction,index)
  mfs = map(_getindex,f.single_fe_functions)
  sff = map(_getindex,f.fe_space)
  fv = f.free_values[index]
  MultiFieldFEFunction(fv,mfs,sff)
end

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

  @assert isa(nodaldata,ParamArray)
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

abstract type FEPFunction end

function _lengths(cellfields::Dict{String,<:FEPFunction})
  lengths = []
  for (k,v) in cellfields
    push!(lengths,length(v))
  end
  L = lengths[1]
  @assert all(lengths .== L)
  return L
end

function Visualization._prepare_pdata(trian,cellfields::Dict{String,<:FEPFunction},samplingpoints)
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
  ParamArray(pdata)
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
