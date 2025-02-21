module ParamGeometry

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Helpers

using GridapEmbedded
using GridapEmbedded.Interfaces
using GridapEmbedded.LevelSetCutters

using ROM.Utils
using ROM.ParamDataStructures

import FillArrays: Fill

export NODES_IN
export NODES_OUT
export TriangulationFilter
export get_nodes_to_cut_mask
include("EmbeddedUtils.jl")

function Geometry.pos_neg_data(
  ipos_to_val::AbstractArray{<:AbstractParamArray{<:Number}},
  i_to_iposneg::PosNegPartition)
  nineg = length(i_to_iposneg.ineg_to_i)
  val = testitem(ipos_to_val)
  zs = 0 .* size(val)
  void = similar(val,eltype(val),zs)
  ineg_to_val = Fill(void,nineg)
  ipos_to_val,ineg_to_val
end

function Geometry._setsize_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._setsize_compress!(ai,bi)
  end
end

function Geometry._uncached_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._uncached_compress!(ai,bi)
  end
  a
end

function Geometry._uncached_compress!(a::AbstractParamArray,b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    Geometry._uncached_compress!(ai,b)
  end
  a
end

function Geometry._copyto_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._copyto_compress!(ai,bi)
  end
  a
end

function Geometry._addto_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._addto_compress!(ai,bi)
  end
  a
end

end
