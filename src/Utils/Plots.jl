mutable struct PlotInfo{T}
  xval::Vector{Vector{T}}
  yval::Vector{Vector{T}}
  title::String
  path::String
  label::Vector{String}
  color::Vector{String}
  style::Vector{String}
  dash::Vector{String}
end

function PlotInfo(
  xval::AbstractArray{T},
  yval::AbstractArray{T},
  title::String,
  path::String,
  label::Vector{String};
  color=["blue"],style=["lines"],dash=[""]) where T

  xvec = vblocks(xval)
  yvec = vblocks(yval)
  @assert length(xvec) == length(xvec) == length(label) "Sizes must be equal"

  function modify_kwarg(kwarg::Vector{String})
    right_len = length(xvec)
    length(kwarg) != right_len ? repeat(first(kwarg),right_len) : kwarg
  end

  new_color = modify_csd(color)
  new_style = modify_csd(style)
  new_dash = modify_csd(dash)

  PlotInfo{T}(xvec,yvec,title,path,label,new_color,new_style,new_dash)
end

function get_layout(pinfo::PlotInfo,::Val{false},::Val{false})
  Layout(title=pinfo.title,xaxis_title=pinfo.xlab,yaxis_title=pinfo.ylab)
end

function get_layout(pinfo::PlotInfo,::Val{true},::Val{false})
  Layout(title=pinfo.title,xaxis_title=pinfo.xlab,yaxis_title=pinfo.ylab,
    xaxis_type="log")
end

function get_layout(pinfo::PlotInfo,::Val{false},::Val{true})
  Layout(title=pinfo.title,xaxis_title=pinfo.xlab,yaxis_title=pinfo.ylab,
    yaxis_type="log")
end

function get_layout(pinfo::PlotInfo,::Val{true},::Val{true})
  Layout(title=pinfo.title,xaxis_title=pinfo.xlab,yaxis_title=pinfo.ylab,
    xaxis_type="log",yaxis_type="log")
end

function PlotlyJS.plot(pinfo::PlotInfo,semilogx::Bool=false,semilogy::Bool=true)
  layout = get_layout(pinfo,Val(semilogx),Val(semilogy))
  pinfo_zip = zip(pinfo.xval,pinfo.yval,pinfo.style,pinfo.label,pinfo.color,pinfo.dash)
  traces = [scatter(x=x,y=y,mode=s,name=l,line=attr(width=4,color=c,dash=d))
    for (x,y,s,l,c,d) in pinfo_zip]
  p = plot(traces,layout)
  savefig(p,pinfo.path*".eps")
end
