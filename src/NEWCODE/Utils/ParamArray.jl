struct ParamArray
  f::Function
end

function Base.:+(pa1::ParamArray,pa2::ParamArray)
  f12 = (args...) -> pa1(args...) + pa2(args...)
  ParamArray(f12)
end

function evaluate(pa::ParamArray,args...)
  pa.f(args...)
end

function evaluate(pa::Vector{ParamArray},args...)
  blocks = pmap(axes(pa,1)) do row
    evaluate(pa[row],args...)
  end
  pa_vec = vcat(blocks...)
  pa_vec
end

function evaluate(pa::Matrix{ParamArray},args...)
  cblocks = pmap(axes(pa,2)) do col
    rblocks = map(axes(pa,1)) do row
      evaluate(pa[row,col],args...)
    end
    vcat(rblocks...)
  end
  pa_mat = hcat(cblocks...)
  pa_mat
end

(pa::ParamArray)(args...) = evaluate(pa,args...)
(pa::Vector{ParamArray})(args...) = evaluate(pa,args...)
(pa::Matrix{ParamArray})(args...) = evaluate(pa,args...)

function Gridap.CellData.add_contribution!(
  pa1::Array{ParamArray,N},
  pa2::ParamArray,
  idx::Int...) where N

  try
    pa1[idx...] += pa2
  catch
    pa1[idx...] = pa2
  end
end
