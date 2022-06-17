"""Plot the surface of the function 'f':R²->R between ranges specificed by
  'xrange' and 'yrange'. The plotting grid is composed of 'n' points/direction"""
function plot_R²_R(f::Function, xrange::Vector, yrange::Vector, n::Int)
  x = range(xrange[1], xrange[2], n)
  y = range(yrange[1], yrange[2], n)
  suface(x, y, f)
end

"""Plot the vector-valued function 'f':R->R² between range specificed by
  'xrange'. The plotting grid is composed of 'n' points"""
function plot_R_R²(f::Function, xrange::Vector, n::Int)
  xs_ys(vs) = Tuple(eltype(vs[1])[vs[i][j] for i in eachindex(vs)]
    for j in eachindex(first(vs)))
  xs_ys(v, vs...) = xs_ys([v, vs...])
  xs_ys(g::Function, a, b, n=100) = xs_ys(g.(range(a, b, n)))
  Plot.plot(xs_ys(f, xrange[1], xrange[2], n)...)
end

function generate_and_save_plot(xval::Array, yval::Array, title::String,
  label::Vector, xlab::String, ylab::String, save_path::String,
  semilogx=false, semilogy=true; var="u",modes=["lines"])

  @assert size(xval) == size(yval) "Invalid plot: provide an input with the same
    x-values as its y-values"
  @assert length(size(xval)) == 2 "Invalid plot: provide an input which is at
    most a matrix"

  if !semilogx && !semilogy
    layout = Layout(title=title,xaxis_title=xlab,yaxis_title=ylab)
  elseif semilogx && !semilogy
    layout = Layout(title=title,xaxis_title=xlab,yaxis_title=ylab,
      xaxis_type="log")
  elseif !semilogx && semilogy
    layout = Layout(title=title,xaxis_title=xlab,yaxis_title=ylab,
      yaxis_type="log")
  else
    layout = Layout(title=title,xaxis_title=xlab,yaxis_title=ylab,
      xaxis_type="log",yaxis_type="log")
  end

  n_traces = size(xval)[1]
  traces = [scatter(x=xval[i,:],y=yval[i,:],mode=modes[i],name=label[i],
    line=attr(width=4)) for i=1:n_traces]
  p = plot(trace,layout)
  savefig(p, joinpath(save_path, string(var)*".eps"))

end
