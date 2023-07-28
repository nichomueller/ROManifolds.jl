struct ParamStructure
  f::Function
end

function Base.:+(pa1::ParamStructure,pa2::ParamStructure)
  f12 = (args...) -> pa1(args...) + pa2(args...)
  ParamStructure(f12)
end

function evaluate(pa::ParamStructure,args...)
  pa.f(args...)
end

function evaluate(pa::Vector{ParamStructure},args...)
  blocks = map(axes(pa,1)) do row
    evaluate(pa[row],args...)
  end
  pa_vec = vcat(blocks...)
  pa_vec
end

function evaluate(pa::Matrix{ParamStructure},args...)
  cblocks = pmap(axes(pa,2)) do col
    rblocks = map(axes(pa,1)) do row
      evaluate(pa[row,col],args...)
    end
    vcat(rblocks...)
  end
  pa_mat = hcat(cblocks...)
  pa_mat
end

(pa::ParamStructure)(args...) = evaluate(pa,args...)
(pa::Vector{ParamStructure})(args...) = evaluate(pa,args...)
(pa::Matrix{ParamStructure})(args...) = evaluate(pa,args...)

function sum_contributions!(
  pa1::Array{ParamStructure,N},
  pa2::ParamStructure,
  idx::Int...) where N

  try
    pa1[idx...] += pa2
  catch
    pa1[idx...] = pa2
  end
end
